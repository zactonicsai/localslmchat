from __future__ import annotations

from pathlib import Path
from typing import Iterator, List


class KeywordLoader:
    """Load keywords from a file using a generator-first approach."""

    def __init__(self, keyword_file: str) -> None:
        self.keyword_file = Path(keyword_file)

    def stream_keywords(self) -> Iterator[str]:
        """Yield one cleaned keyword at a time."""
        with self.keyword_file.open("r", encoding="utf-8") as file:
            for line in file:
                keyword = line.strip().lower()
                if keyword:
                    yield keyword

    def load_keywords(self) -> List[str]:
        """Return all keywords as a list."""
        return list(self.stream_keywords())
