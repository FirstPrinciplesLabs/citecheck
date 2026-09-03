"""String-similarity helpers used by the cascade.

Two metrics live here:

- :func:`title_similarity` -- Levenshtein-based 0-100 score, used as the
  cascade's *accept/reject* gate (compared against
  ``min_title_similarity``).
- :func:`word_overlap_similarity` -- Jaccard-style 0-100 score, used
  only as a cheap *ranking* heuristic when an API returns multiple
  candidates and we need to pick the best.

Both metrics lowercase and strip punctuation before comparing so that
trivial formatting differences ("Black-hole entropy." vs. "Black hole
entropy") don't leak into the score.
"""

from __future__ import annotations

import re


__all__ = (
    "levenshtein",
    "title_similarity",
    "word_overlap_similarity",
    "clean_query",
)


_PUNCT_RE = re.compile(r"[^\w\s]")


def levenshtein(a: str, b: str) -> int:
    """Levenshtein (edit) distance between two strings.

    Uses an optimised two-row dynamic-programming approach (O(n) extra
    space) instead of a full m*n matrix.  The edit distance counts the
    minimum number of single-character insertions, deletions, or
    substitutions needed to transform *a* into *b*.
    """
    m, n = len(a), len(b)
    prev = list(range(n + 1))
    curr = [0] * (n + 1)
    for i in range(1, m + 1):
        curr[0] = i
        for j in range(1, n + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            curr[j] = min(prev[j] + 1, curr[j - 1] + 1, prev[j - 1] + cost)
        prev, curr = curr, prev
    return prev[n]


def _normalise(s: str) -> str:
    return _PUNCT_RE.sub("", s.lower()).strip()


def title_similarity(a: str, b: str) -> float:
    """Return a 0-100 similarity score between two title strings.

    Lowercases both strings, strips punctuation, then computes::

        score = max(0, (1 - levenshtein(a, b) / max(len(a), len(b))) * 100)

    100 means identical after normalisation, 0 means no characters in
    common.  This is the metric the cascade uses to decide whether a
    candidate from an API actually corresponds to the citation's
    intended paper.
    """
    if not a or not b:
        return 0.0
    ca, cb = _normalise(a), _normalise(b)
    if ca == cb:
        return 100.0
    max_len = max(len(ca), len(cb))
    if max_len == 0:
        return 0.0
    dist = levenshtein(ca, cb)
    return max(0.0, round((1 - dist / max_len) * 100, 1))


def word_overlap_similarity(a: str, b: str) -> float:
    """Return a 0-100 word-overlap similarity score (Jaccard-style).

    Lowercases, strips punctuation, splits on whitespace, then::

        score = |words_a & words_b| / max(|words_a|, |words_b|) * 100

    Cheaper than Levenshtein and used by the Semantic Scholar /
    OpenAlex selectors as a fast ranking heuristic when picking the
    best of multiple API candidates.
    """
    wa = set(_PUNCT_RE.sub("", a.lower()).split())
    wb = set(_PUNCT_RE.sub("", b.lower()).split())
    if not wa or not wb:
        return 0.0
    overlap = len(wa & wb)
    return round(overlap / max(len(wa), len(wb)) * 100, 1)


def clean_query(text: str) -> str:
    """Strip LaTeX / BibTeX artifacts from a query string.

    Removes curly braces ``{}``, the formatting commands ``\\textbf``,
    ``\\textit``, ``\\emph``, ``\\textrm``, escaped ampersands ``\\&``,
    non-breaking tildes ``~``, and collapses runs of whitespace.  Used
    to clean up citation text before sending it to an external API
    that won't understand the markup.
    """
    text = re.sub(r"[{}]", "", text)
    text = re.sub(r"\\textbf|\\textit|\\emph|\\textrm", "", text)
    text = re.sub(r"\\&", "&", text)
    text = re.sub(r"~", " ", text)
    return re.sub(r"\s{2,}", " ", text).strip()
