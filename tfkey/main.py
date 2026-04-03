#!/usr/bin/env python3
"""
main.py

Read a text document from disk and pass its content through the TF-IDF
keyword analyser.  Prints the top-5 electronics / radio-frequency
keywords found (or a clear "none found" message).

Usage
-----
    python main.py <path_to_text_file>
    python main.py sample_document.txt
    python main.py --chunk-size 2048 large_file.txt
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from tfidf_analyzer import find_top_keywords, load_keywords


# ── streaming reader ─────────────────────────────────────────────────

def stream_file(filepath: Path, chunk_size: int = 4096) -> str:
    """Read *filepath* in chunks and return the full text.

    This simulates a streaming / chunked-read pattern so that very
    large files do not need to be slurped into memory in a single call.
    Each chunk is yielded internally and concatenated, but the same
    approach could feed a rolling TF-IDF pipeline.

    Parameters
    ----------
    filepath : Path
        Path to the input text file.
    chunk_size : int
        Bytes per read (default 4 096).

    Returns
    -------
    str
        The complete document text.

    Raises
    ------
    FileNotFoundError
        If *filepath* does not exist.
    """
    if not filepath.exists():
        raise FileNotFoundError(f"File not found: {filepath}")

    chunks: list[str] = []
    bytes_read = 0

    with filepath.open("r", encoding="utf-8") as fh:
        while True:
            chunk = fh.read(chunk_size)
            if not chunk:
                break
            chunks.append(chunk)
            bytes_read += len(chunk.encode("utf-8"))
            print(f"  ... streamed {bytes_read:,} bytes", end="\r", file=sys.stderr)

    print(f"  ✓ finished reading {bytes_read:,} bytes total", file=sys.stderr)
    return "".join(chunks)


# ── CLI ──────────────────────────────────────────────────────────────

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="TF-IDF keyword detector for electronics & RF documents."
    )
    parser.add_argument(
        "file",
        type=Path,
        help="Path to the text document to analyse.",
    )
    parser.add_argument(
        "--keywords-json",
        type=Path,
        default=None,
        help="Path to the keywords JSON file (default: keywords.json next to tfidf_analyzer.py).",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=5,
        help="Number of top keywords to return (default: 5).",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=4096,
        help="Chunk size in bytes for streaming reads (default: 4096).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    filepath: Path = args.file
    print(f"\n{'='*60}")
    print(f"  TF-IDF Keyword Analyser — Electronics & RF")
    print(f"{'='*60}")
    print(f"  Document : {filepath}")
    print()

    # --- stream the file in ---
    try:
        text = stream_file(filepath, chunk_size=args.chunk_size)
    except FileNotFoundError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1

    # --- load keywords and analyse ---
    keywords = load_keywords(args.keywords_json)
    print(f"  Loaded {len(keywords)} domain keywords.\n")

    results = find_top_keywords(
        text,
        keywords=keywords,
        top_n=args.top_n,
    )

    # --- display results ---
    if not results:
        print("  ⚠  No matching keywords found in this document.")
        print("     The text may not contain electronics / RF terminology.")
    else:
        print(f"  Top {len(results)} keyword(s) by TF-IDF score:\n")
        print(f"  {'Rank':<6} {'Keyword':<25} {'Score':<12}")
        print(f"  {'─'*6} {'─'*25} {'─'*12}")
        for rank, entry in enumerate(results, start=1):
            print(f"  {rank:<6} {entry['keyword']:<25} {entry['score']:<12.6f}")

    print(f"\n{'='*60}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
