from pathlib import Path

from app import stream_corpus_files


def test_stream_corpus_files_only_yields_txt_files(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("hello", encoding="utf-8")
    (tmp_path / "b.md").write_text("ignore", encoding="utf-8")
    (tmp_path / "c.txt").write_text("world", encoding="utf-8")

    result = list(stream_corpus_files(str(tmp_path)))

    assert result == [str(tmp_path / "a.txt"), str(tmp_path / "c.txt")]
