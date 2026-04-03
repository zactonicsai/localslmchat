"""
tfidf_analyzer.py

TF-IDF keyword extraction module. Scores document text against a static
list of known electronics / radio-frequency keywords and returns the top
matches (up to five), or an empty list when nothing relevant is found.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from pathlib import Path
from typing import Optional


# ── helpers ──────────────────────────────────────────────────────────

_TOKEN_RE = re.compile(r"[a-z][a-z0-9\-]*(?:\s[a-z][a-z0-9\-]*)?", re.IGNORECASE)


def _tokenize(text: str) -> list[str]:
    """Return lowercased tokens from *text*.

    Handles single words and also captures two-word tokens so that
    multi-word keywords like "integrated circuit" or "power supply" can
    be matched.
    """
    words = text.lower().split()
    # single-word tokens
    tokens = [re.sub(r"[^a-z0-9\-]", "", w) for w in words]
    tokens = [t for t in tokens if t]
    # bigram tokens (for multi-word keywords)
    bigrams = [f"{tokens[i]} {tokens[i+1]}" for i in range(len(tokens) - 1)]
    return tokens + bigrams


def load_keywords(json_path: Optional[str | Path] = None) -> list[str]:
    """Load the keyword list from a JSON file.

    Parameters
    ----------
    json_path : path-like, optional
        Defaults to ``keywords.json`` in the same directory as this module.

    Returns
    -------
    list[str]
        Lowercased keyword strings.
    """
    if json_path is None:
        json_path = Path(__file__).parent / "keywords.json"
    else:
        json_path = Path(json_path)

    with json_path.open("r", encoding="utf-8") as fh:
        data = json.load(fh)

    return [kw.lower() for kw in data["keywords"]]


# ── core TF-IDF logic ───────────────────────────────────────────────

def compute_tf(tokens: list[str], keywords: list[str]) -> dict[str, float]:
    """Compute term-frequency for each keyword found in *tokens*.

    TF = (count of keyword in document) / (total tokens in document)

    Only keywords that actually appear are included in the result.
    """
    total = len(tokens) if tokens else 1
    counts = Counter(tokens)
    return {kw: counts[kw] / total for kw in keywords if counts.get(kw, 0) > 0}


def compute_idf(keywords: list[str], corpus_size: int = 1) -> dict[str, float]:
    """Return a synthetic IDF weight for each keyword.

    With a single document there is no true corpus, so we use keyword
    *specificity* as a proxy: shorter / more common English stems get a
    lower IDF while longer, more technical terms get a higher one.

    The formula used is::

        idf = 1 + log(1 + len(keyword) / avg_len)

    This rewards domain-specific multi-word terms (e.g. "integrated
    circuit") over generic short ones (e.g. "gain").
    """
    if not keywords:
        return {}
    avg_len = sum(len(kw) for kw in keywords) / len(keywords)
    return {kw: 1.0 + math.log(1.0 + len(kw) / avg_len) for kw in keywords}


def find_top_keywords(
    text: str,
    keywords: Optional[list[str]] = None,
    json_path: Optional[str | Path] = None,
    top_n: int = 5,
    min_score: float = 0.0,
) -> list[dict[str, float]]:
    """Analyse *text* with TF-IDF and return the top matching keywords.

    Parameters
    ----------
    text : str
        The document body to analyse.
    keywords : list[str], optional
        Pre-loaded keyword list. If *None*, loaded from *json_path*.
    json_path : path-like, optional
        Path to the keyword JSON file (used only when *keywords* is None).
    top_n : int
        Maximum number of results to return (default 5).
    min_score : float
        Minimum TF-IDF score to include a keyword (default 0.0 — must
        appear at least once).

    Returns
    -------
    list[dict]
        Each dict has ``{"keyword": str, "score": float}`` sorted by
        descending score.  Returns an empty list when no keywords are
        found in the document.
    """
    if keywords is None:
        keywords = load_keywords(json_path)

    tokens = _tokenize(text)
    if not tokens:
        return []

    tf = compute_tf(tokens, keywords)
    if not tf:
        return []

    idf = compute_idf(keywords)

    scored = {kw: tf[kw] * idf.get(kw, 1.0) for kw in tf}

    ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)

    results = [
        {"keyword": kw, "score": round(score, 6)}
        for kw, score in ranked
        if score > min_score
    ]

    return results[:top_n]
