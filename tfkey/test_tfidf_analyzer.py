"""
test_tfidf_analyzer.py

Pytest suite covering:
  - keyword loading
  - tokenisation
  - TF / IDF computation
  - end-to-end keyword extraction (top-5 and none-found)
  - main.py CLI integration
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

# ── locate project root so imports work when running from any cwd ────
PROJECT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_DIR))

from tfidf_analyzer import (
    _tokenize,
    compute_idf,
    compute_tf,
    find_top_keywords,
    load_keywords,
)


# ── fixtures ─────────────────────────────────────────────────────────

@pytest.fixture
def keywords_path() -> Path:
    return PROJECT_DIR / "keywords.json"


@pytest.fixture
def sample_doc_path() -> Path:
    return PROJECT_DIR / "sample_document.txt"


@pytest.fixture
def no_match_doc_path() -> Path:
    return PROJECT_DIR / "sample_no_match.txt"


@pytest.fixture
def keywords_list(keywords_path) -> list[str]:
    return load_keywords(keywords_path)


@pytest.fixture
def sample_text(sample_doc_path) -> str:
    return sample_doc_path.read_text(encoding="utf-8")


@pytest.fixture
def no_match_text(no_match_doc_path) -> str:
    return no_match_doc_path.read_text(encoding="utf-8")


# ── keyword loading ──────────────────────────────────────────────────

class TestLoadKeywords:
    def test_loads_list(self, keywords_list):
        assert isinstance(keywords_list, list)
        assert len(keywords_list) > 50  # we have ~120 keywords

    def test_all_lowercase(self, keywords_list):
        for kw in keywords_list:
            assert kw == kw.lower(), f"Keyword not lowercase: {kw!r}"

    def test_known_keywords_present(self, keywords_list):
        for expected in ["antenna", "frequency", "transistor", "impedance"]:
            assert expected in keywords_list

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_keywords(tmp_path / "nonexistent.json")


# ── tokenisation ─────────────────────────────────────────────────────

class TestTokenize:
    def test_basic(self):
        tokens = _tokenize("The quick brown fox")
        assert "the" in tokens
        assert "quick" in tokens

    def test_lowercases(self):
        tokens = _tokenize("ANTENNA Signal")
        assert "antenna" in tokens
        assert "signal" in tokens

    def test_bigrams_generated(self):
        tokens = _tokenize("integrated circuit design")
        assert "integrated circuit" in tokens
        assert "circuit design" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []

    def test_punctuation_stripped(self):
        tokens = _tokenize("resistor, capacitor; inductor.")
        assert "resistor" in tokens
        assert "capacitor" in tokens
        assert "inductor" in tokens


# ── term frequency ───────────────────────────────────────────────────

class TestComputeTF:
    def test_single_keyword(self):
        tokens = ["antenna", "the", "big", "antenna"]
        tf = compute_tf(tokens, ["antenna"])
        assert tf["antenna"] == pytest.approx(2 / 4)

    def test_missing_keyword_excluded(self):
        tokens = ["antenna", "signal"]
        tf = compute_tf(tokens, ["transistor"])
        assert tf == {}

    def test_multiple_keywords(self):
        tokens = ["antenna", "signal", "antenna", "noise"]
        tf = compute_tf(tokens, ["antenna", "noise", "capacitor"])
        assert "antenna" in tf
        assert "noise" in tf
        assert "capacitor" not in tf

    def test_empty_tokens(self):
        tf = compute_tf([], ["antenna"])
        assert tf == {}


# ── IDF weights ──────────────────────────────────────────────────────

class TestComputeIDF:
    def test_longer_keywords_get_higher_weight(self):
        idf = compute_idf(["am", "superheterodyne"])
        assert idf["superheterodyne"] > idf["am"]

    def test_empty_list(self):
        assert compute_idf([]) == {}

    def test_all_positive(self):
        idf = compute_idf(["antenna", "fm", "oscillator"])
        assert all(v > 0 for v in idf.values())


# ── end-to-end: find_top_keywords ───────────────────────────────────

class TestFindTopKeywords:
    def test_returns_up_to_five(self, sample_text, keywords_list):
        results = find_top_keywords(sample_text, keywords=keywords_list, top_n=5)
        assert 1 <= len(results) <= 5

    def test_result_structure(self, sample_text, keywords_list):
        results = find_top_keywords(sample_text, keywords=keywords_list)
        for entry in results:
            assert "keyword" in entry
            assert "score" in entry
            assert isinstance(entry["keyword"], str)
            assert isinstance(entry["score"], float)

    def test_scores_descending(self, sample_text, keywords_list):
        results = find_top_keywords(sample_text, keywords=keywords_list)
        scores = [r["score"] for r in results]
        assert scores == sorted(scores, reverse=True)

    def test_known_keywords_surface(self, sample_text, keywords_list):
        results = find_top_keywords(sample_text, keywords=keywords_list, top_n=20)
        found = {r["keyword"] for r in results}
        # The sample doc heavily references these terms
        expected_subset = {"antenna", "frequency", "signal", "impedance"}
        assert expected_subset.issubset(found), (
            f"Missing expected keywords: {expected_subset - found}"
        )

    def test_no_match_returns_empty(self, no_match_text, keywords_list):
        results = find_top_keywords(no_match_text, keywords=keywords_list)
        assert results == []

    def test_empty_text_returns_empty(self, keywords_list):
        results = find_top_keywords("", keywords=keywords_list)
        assert results == []

    def test_top_n_respected(self, sample_text, keywords_list):
        for n in (1, 3, 5):
            results = find_top_keywords(sample_text, keywords=keywords_list, top_n=n)
            assert len(results) <= n

    def test_loads_keywords_from_json(self, sample_text, keywords_path):
        """Verify the json_path fallback works without pre-loading."""
        results = find_top_keywords(sample_text, json_path=keywords_path, top_n=5)
        assert len(results) >= 1

    def test_custom_keyword_list(self):
        text = "banana banana apple apple apple cherry"
        results = find_top_keywords(text, keywords=["apple", "banana", "cherry"])
        assert results[0]["keyword"] == "apple"

    def test_multiword_keyword_match(self):
        text = "The integrated circuit is a key part of the integrated circuit design."
        results = find_top_keywords(text, keywords=["integrated circuit"])
        assert len(results) == 1
        assert results[0]["keyword"] == "integrated circuit"


# ── main.py CLI integration ──────────────────────────────────────────

class TestMainCLI:
    def _run(self, *args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, str(PROJECT_DIR / "main.py"), *args],
            capture_output=True,
            text=True,
        )

    def test_sample_doc_succeeds(self, sample_doc_path):
        result = self._run(str(sample_doc_path))
        assert result.returncode == 0
        assert "Top" in result.stdout
        assert "keyword" not in result.stdout or "Keyword" in result.stdout

    def test_no_match_doc_message(self, no_match_doc_path):
        result = self._run(str(no_match_doc_path))
        assert result.returncode == 0
        assert "No matching keywords found" in result.stdout

    def test_missing_file_exits_nonzero(self, tmp_path):
        result = self._run(str(tmp_path / "does_not_exist.txt"))
        assert result.returncode == 1
        assert "not found" in result.stderr.lower() or "not found" in result.stdout.lower()

    def test_top_n_flag(self, sample_doc_path):
        result = self._run("--top-n", "2", str(sample_doc_path))
        assert result.returncode == 0
        # Count the numbered result lines (they start with whitespace + digit)
        lines = [l for l in result.stdout.splitlines() if l.strip() and l.strip()[0].isdigit()]
        assert len(lines) <= 2
