from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterator, List

from keyword_loader import KeywordLoader
from tfidf_ranker import TfidfKeywordRanker
from topn_ranker import TopNKeywordSelector


def stream_corpus_files(corpus_dir: str) -> Iterator[str]:
    """Yield corpus file paths one at a time."""
    path = Path(corpus_dir)
    for file_path in sorted(path.iterdir()):
        if file_path.is_file() and file_path.suffix.lower() == ".txt":
            yield str(file_path)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rank known keywords in a document using memory-efficient TF-IDF."
    )
    parser.add_argument("--keywords", required=True, help="Path to keywords.txt")
    parser.add_argument("--document", required=True, help="Path to target document")
    parser.add_argument("--corpus-dir", required=True, help="Directory containing corpus .txt files")
    parser.add_argument("--top-n", type=int, default=5, help="Number of top keywords to return")
    args = parser.parse_args()

    keyword_loader = KeywordLoader(args.keywords)
    keywords = keyword_loader.load_keywords()

    ranker = TfidfKeywordRanker(keywords)
    selector = TopNKeywordSelector()

    corpus_files: List[str] = list(stream_corpus_files(args.corpus_dir))
    keyword_scores = ranker.rank_keywords(args.document, corpus_files)
    top_keywords = selector.get_top_n(keyword_scores, args.top_n)

    print("\nTop ranked keywords:")
    print("-" * 40)
    for keyword, score in top_keywords:
        print(f"{keyword:20} -> {score:.6f}")


if __name__ == "__main__":
    main()
