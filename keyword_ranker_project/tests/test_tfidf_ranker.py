from pathlib import Path

import pytest

from tfidf_ranker import TfidfKeywordRanker


def create_text_file(path: Path, content: str) -> str:
    path.write_text(content, encoding="utf-8")
    return str(path)


def test_init_raises_for_empty_keywords() -> None:
    with pytest.raises(ValueError):
        TfidfKeywordRanker([])


def test_stream_document_tokens(tmp_path: Path) -> None:
    doc = create_text_file(tmp_path / "doc.txt", "Python cloud-based ranking")
    ranker = TfidfKeywordRanker(["python", "cloud", "ranking"])

    tokens = list(ranker.stream_document_tokens(doc))

    assert tokens == ["python", "cloud-based", "ranking"]


def test_compute_tf_scores(tmp_path: Path) -> None:
    doc = create_text_file(
        tmp_path / "doc.txt",
        "Python python cloud ranking machine learning machine learning"
    )
    ranker = TfidfKeywordRanker(["python", "cloud", "ranking", "machine learning"])

    tf = ranker.compute_tf(doc)

    assert tf["python"] > tf["cloud"]
    assert tf["machine learning"] > 0.0
    assert tf["ranking"] > 0.0


def test_compute_idf_and_rank_keywords(tmp_path: Path) -> None:
    doc = create_text_file(
        tmp_path / "target.txt",
        "Python python machine learning ranking document"
    )
    corpus1 = create_text_file(tmp_path / "c1.txt", "python ranking")
    corpus2 = create_text_file(tmp_path / "c2.txt", "cloud security document")
    corpus3 = create_text_file(tmp_path / "c3.txt", "machine learning nlp")

    keywords = ["python", "machine learning", "ranking", "document", "security"]
    ranker = TfidfKeywordRanker(keywords)

    scores = ranker.rank_keywords(doc, [corpus1, corpus2, corpus3])

    assert set(scores.keys()) == set(keywords)
    assert all(0.0 <= value <= 1.0 for value in scores.values())
    assert max(scores.values()) == 1.0
    assert scores["security"] == 0.0
