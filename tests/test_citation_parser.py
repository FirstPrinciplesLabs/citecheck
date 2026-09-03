"""Citation parser smoke tests."""

from citecheck.core import parse_citation


def test_parse_markdown_link_citation():
    raw = (
        "[Vaswani et al., 2017, Attention Is All You Need]"
        "(https://arxiv.org/abs/1706.03762)"
    )
    citation = parse_citation(1, raw)

    assert citation.number == 1
    assert citation.authors == "Vaswani et al."
    assert citation.year == 2017
    assert citation.title == "Attention Is All You Need"
    assert citation.url == "https://arxiv.org/abs/1706.03762"
    assert citation.arxiv_id == "1706.03762"


def test_parse_plain_text_citation_without_url():
    raw = "Doe, J., 2023, A Paper Without a URL"
    citation = parse_citation(2, raw)

    assert citation.authors == "Doe"
    assert citation.year == 2023
    assert citation.title == "A Paper Without a URL"
    assert citation.url is None
    assert citation.arxiv_id is None
