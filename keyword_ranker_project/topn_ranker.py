from __future__ import annotations

from typing import Dict, List, Tuple


class TopNKeywordSelector:
    """Select the top N keywords from a score dictionary."""

    def get_top_n(self, keyword_scores: Dict[str, float], top_n: int) -> List[Tuple[str, float]]:
        if top_n <= 0:
            raise ValueError("top_n must be greater than 0")

        ranked = sorted(keyword_scores.items(), key=lambda item: item[1], reverse=True)
        return ranked[:top_n]
