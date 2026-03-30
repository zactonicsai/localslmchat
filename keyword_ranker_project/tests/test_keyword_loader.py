from pathlib import Path

from keyword_loader import KeywordLoader


def test_stream_keywords_yields_clean_lowercase_keywords(tmp_path: Path) -> None:
    keyword_file = tmp_path / "keywords.txt"
    keyword_file.write_text(" Python\n\nMachine Learning\nTF-IDF\n", encoding="utf-8")

    loader = KeywordLoader(str(keyword_file))

    assert list(loader.stream_keywords()) == ["python", "machine learning", "tf-idf"]


def test_load_keywords_returns_list(tmp_path: Path) -> None:
    keyword_file = tmp_path / "keywords.txt"
    keyword_file.write_text("python\ncloud\n", encoding="utf-8")

    loader = KeywordLoader(str(keyword_file))

    assert loader.load_keywords() == ["python", "cloud"]
