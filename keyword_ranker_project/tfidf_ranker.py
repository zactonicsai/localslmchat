from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Dict, Iterator, List


class TfidfKeywordRanker:
    """
    Rank known keywords for a target document using a simple streaming TF-IDF approach.

    Notes:
    - Streams the target document line by line.
    - Uses yield-based token generation to reduce memory footprint.
    - Keeps keyword scoring normalized to 0..1.
    """

    WORD_PATTERN = re.compile(r"\b[a-zA-Z0-9][a-zA-Z0-9\-]*\b")

    def __init__(self, keywords: List[str]) -> None:
        if not keywords:
            raise ValueError("keywords cannot be empty")

        self.keywords = [keyword.strip().lower() for keyword in keywords if keyword.strip()]
        if not self.keywords:
            raise ValueError("keywords cannot be empty after cleaning")

        self.keyword_token_map = {
            keyword: self._tokenize_phrase(keyword) for keyword in self.keywords
        }

    def _tokenize_phrase(self, text: str) -> List[str]:
        return [token.lower() for token in self.WORD_PATTERN.findall(text.lower())]

    def stream_document_lines(self, file_path: str) -> Iterator[str]:
        with Path(file_path).open("r", encoding="utf-8") as file:
            for line in file:
                yield line

    def stream_document_tokens(self, file_path: str) -> Iterator[str]:
        for line in self.stream_document_lines(file_path):
            for token in self.WORD_PATTERN.findall(line.lower()):
                yield token

    def count_terms_in_document(self, file_path: str) -> Counter:
        counts: Counter = Counter()
        for token in self.stream_document_tokens(file_path):
            counts[token] += 1
        return counts

    def compute_document_frequency(self, corpus_files: List[str]) -> Dict[str, int]:
        doc_freq: Dict[str, int] = {keyword: 0 for keyword in self.keywords}

        for corpus_file in corpus_files:
            token_set = set(self.stream_document_tokens(corpus_file))

            for keyword, parts in self.keyword_token_map.items():
                if not parts:
                    continue

                if len(parts) == 1:
                    if parts[0] in token_set:
                        doc_freq[keyword] += 1
                else:
                    doc_tokens = list(self.stream_document_tokens(corpus_file))
                    joined_text = " ".join(doc_tokens)
                    phrase = " ".join(parts)
                    if phrase in joined_text:
                        doc_freq[keyword] += 1

        return doc_freq

    def compute_idf(self, corpus_files: List[str]) -> Dict[str, float]:
        total_docs = len(corpus_files)
        if total_docs == 0:
            return {keyword: 1.0 for keyword in self.keywords}

        doc_freq = self.compute_document_frequency(corpus_files)
        return {
            keyword: math.log((total_docs + 1) / (df + 1)) + 1.0
            for keyword, df in doc_freq.items()
        }

    def compute_tf(self, document_file: str) -> Dict[str, float]:
        term_counts = self.count_terms_in_document(document_file)
        total_terms = sum(term_counts.values())

        if total_terms == 0:
            return {keyword: 0.0 for keyword in self.keywords}

        tf_scores: Dict[str, float] = {}

        if any(len(parts) > 1 for parts in self.keyword_token_map.values()):
            doc_tokens = list(self.stream_document_tokens(document_file))
            joined_text = " ".join(doc_tokens)
        else:
            joined_text = ""

        for keyword, parts in self.keyword_token_map.items():
            if not parts:
                tf_scores[keyword] = 0.0
                continue

            if len(parts) == 1:
                raw_count = term_counts.get(parts[0], 0)
            else:
                phrase = " ".join(parts)
                raw_count = joined_text.count(phrase)

            tf_scores[keyword] = raw_count / total_terms

        return tf_scores

    def rank_keywords(self, document_file: str, corpus_files: List[str]) -> Dict[str, float]:
        tf_scores = self.compute_tf(document_file)
        idf_scores = self.compute_idf(corpus_files)

        raw_scores = {
            keyword: tf_scores[keyword] * idf_scores[keyword]
            for keyword in self.keywords
        }

        max_score = max(raw_scores.values(), default=0.0)
        if max_score == 0.0:
            return {keyword: 0.0 for keyword in self.keywords}

        return {
            keyword: round(score / max_score, 6)
            for keyword, score in raw_scores.items()
        }
