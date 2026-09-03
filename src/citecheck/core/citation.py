"""Citation parsing.

A :class:`Citation` carries the structured fields the cascade and the
LLM verifier need (authors, year, title, URL, arXiv ID).
:func:`parse_citation` turns a raw citation string into a
:class:`Citation`, handling three surface forms:

1. ``[Authors, Year, Title](URL)`` -- markdown-link format used by
   the bundled dataset.
2. ``Authors, Year, Title (URL)``  -- plain parenthesised URL.
3. ``Authors, Year, Title``        -- no URL at all.

This module deliberately knows *nothing* about the dataset schema;
that lives in :mod:`citecheck.eval.dataset`.  The same
:func:`parse_citation` function is used by every downstream consumer.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Optional


# ---------------------------------------------------------------------------
# Citation dataclass
# ---------------------------------------------------------------------------

@dataclass
class Citation:
    """Structured representation of a single citation."""

    number: int
    raw_text: str
    authors: Optional[str] = None
    year: Optional[int] = None
    title: Optional[str] = None
    url: Optional[str] = None
    arxiv_id: Optional[str] = None


# ---------------------------------------------------------------------------
# arXiv ID extraction
# ---------------------------------------------------------------------------

_ARXIV_PATTERNS = (
    r"arxiv\.org/abs/([^\s\)]+)",
    r"arxiv\.org/pdf/([^\s\)]+?)\.pdf",
    r"arxiv\.org/pdf/([^\s\)]+)",
    r"arxiv\.org/html/([^\s\)]+)",
    r"arXiv:(\S+)",
)


def extract_arxiv_id(text: str) -> Optional[str]:
    """Return the arXiv ID found in *text*, or ``None``.

    Handles a wide range of formats:

    - ``https://arxiv.org/abs/2507.00504``
    - ``https://arxiv.org/pdf/2107.09676.pdf``
    - ``https://arxiv.org/html/2508.19075``
    - ``arXiv:2508.19075``
    - ``https://arxiv.org/abs/quant-ph/0112031``  (old-style IDs)
    - IDs with version suffixes like ``2107.09676v2``
    """
    for pattern in _ARXIV_PATTERNS:
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip().rstrip(")")
    return None


# ---------------------------------------------------------------------------
# Single-citation parser
# ---------------------------------------------------------------------------

# Markdown link wrapper:  [Authors, Year, Title](URL)
# Trailing text after the link (e.g. "... (background)") is tolerated.
_MD_LINK_RE = re.compile(
    r"^\s*\[(?P<inside>.+?)\]\((?P<url>https?://[^\s)]+)\)",
    re.DOTALL,
)

# Plain parenthesised URL:  ... (URL)
_PAREN_URL_RE = re.compile(r"\((?P<url>https?://[^\s)]+)\)")

# Year: a four-digit number whose century is 19xx or 20xx.
_YEAR_RE = re.compile(r"\b(?:19|20)\d{2}\b")

# Authors: everything before the first comma in the cleaned inner text.
_AUTHORS_RE = re.compile(r"^([^,]+),")

# Title: everything after the *first* "YYYY, " in the inner text.  Matches
# both the canonical "Authors, YYYY, Title" form and the
# "Authors, Initial. YYYY, Title" form that minor-corruption rewrites
# can introduce (where the year is no longer comma-prefixed but is
# still comma-suffixed before the title).
_TITLE_RE = re.compile(r"(?:19|20)\d{2}\.?\s*,\s*(?P<title>.+?)\s*$", re.DOTALL)


def _peel_markdown_link(text: str) -> tuple[str, Optional[str]]:
    """Return ``(inner, url)`` if *text* is a markdown link, else ``(text, None)``.

    ``inner`` is the bracketed content with the surrounding ``[`` ``]``
    stripped; ``url`` is the link target.
    """
    m = _MD_LINK_RE.match(text)
    if m:
        return m.group("inside").strip(), m.group("url").strip()
    return text, None


def parse_citation(citation_number: int, citation_text: str) -> Citation:
    """Parse a raw citation string into a :class:`Citation`.

    See the module docstring for the surface forms supported.  Fields
    that cannot be extracted remain ``None`` so the caller can decide
    whether to invoke an LLM-based fallback parser.
    """
    citation = Citation(number=citation_number, raw_text=citation_text)

    inner, url_from_brackets = _peel_markdown_link(citation_text)

    if url_from_brackets is not None:
        citation.url = url_from_brackets
    else:
        url_match = _PAREN_URL_RE.search(citation_text)
        if url_match:
            citation.url = url_match.group("url")

    citation.arxiv_id = extract_arxiv_id(citation_text)

    year_match = _YEAR_RE.search(inner)
    if year_match:
        citation.year = int(year_match.group(0))

    author_match = _AUTHORS_RE.match(inner)
    if author_match:
        citation.authors = author_match.group(1).strip()

    title_match = _TITLE_RE.search(inner)
    if title_match:
        candidate = title_match.group("title").strip()
        # Trim a trailing ")" left over from nested-paren artifacts.
        if candidate.endswith(")") and candidate.count("(") < candidate.count(")"):
            candidate = candidate[:-1].rstrip()
        citation.title = candidate
    elif citation.year is None and citation.authors is None and inner:
        # Title-only form: "[Some title](URL)" with no authors/year.
        # Fall back to using the bracketed inner text as the title so the
        # LLM verifier has something to compare against.
        citation.title = inner

    return citation
