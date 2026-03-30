import pytest

from topn_ranker import TopNKeywordSelector


def test_get_top_n_returns_expected_items() -> None:
    selector = TopNKeywordSelector()
    scores = {"python": 0.8, "cloud": 0.2, "nlp": 0.6}

    result = selector.get_top_n(scores, 2)

    assert result == [("python", 0.8), ("nlp", 0.6)]


def test_get_top_n_raises_for_invalid_n() -> None:
    selector = TopNKeywordSelector()

    with pytest.raises(ValueError):
        selector.get_top_n({"python": 1.0}, 0)
