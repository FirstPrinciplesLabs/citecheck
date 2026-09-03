"""Prompt templates for the LLM-only baseline harness.

Three pieces are exported:

- :data:`BASELINE_SYSTEM_PROMPT` — the rubric-bearing system prompt,
  used for both zero-shot and few-shot.
- :data:`WEB_SEARCH_NUDGE` — short paragraph appended to the system
  prompt when web search is enabled, telling the model to actually use
  the tool and how to handle ``paper not found``.
- :func:`build_user_prompt(citation_text, exemplars=None)` — produces
  the user message for either zero-shot (no exemplars) or few-shot
  (exemplars rendered as a numbered list before the target citation).

Design notes
------------
The rubric mirrors the verifier's prompt
(:mod:`citecheck.llm.prompts`) so that the only varying axis between
baselines and CiteCheck's main detector is *what information the model
has access to* — not the definition of the labels themselves.  The
0-10 score scale is preserved so calibrated thresholds could be
applied the same way the main detector's
:func:`~citecheck.core.classify.classify_score` does.

Expected output schema (a single JSON object with four fields)::

    {
        "label":          "exact_match" | "minor_hallucination" | "major_hallucination",
        "score":          <number 0-10>,
        "reasoning":      "<short justification>",
        "found_evidence": <true | false>
    }

``found_evidence`` is mainly informative for the web-search variants:
it lets us track how often each model says "I couldn't find this
paper", which we then map to ``major_hallucination`` per project
spec.
"""

from __future__ import annotations

from typing import Iterable, Optional

from ..eval.splits import Exemplar


__all__ = (
    "BASELINE_SYSTEM_PROMPT",
    "WEB_SEARCH_NUDGE",
    "build_system_prompt",
    "build_user_prompt",
)


# ---------------------------------------------------------------------------
# System prompt (rubric)
# ---------------------------------------------------------------------------

BASELINE_SYSTEM_PROMPT = """\
You are a citation verification expert. You will be shown a single citation that appears in a research report written by an LLM. Your job is to decide whether the citation is real and accurate, has minor errors, or is a fabrication.

**Three classes:**

- **exact_match**: The citation correctly identifies a real paper. Authors, year, title, and any identifiers (URL, arXiv ID, DOI) all match the actual paper, possibly with trivial formatting differences (e.g. "et al.", title truncation, punctuation).

- **minor_hallucination**: The citation refers to a recognisable real paper but has small metadata errors. Examples:
  - Author name variations (added/removed initials, slight misspellings)
  - Year off by 1-2 years
  - Minor title word changes (1-2 words, synonyms)
  - arXiv ID off by 1-2 digits
  - URL pointing to a different mirror but the same paper
  The core identity of the paper is still clear.

- **major_hallucination**: The citation describes a different paper from what its identifiers point to, or the paper does not exist at all. Examples:
  - Completely fabricated paper that cannot be found anywhere
  - Authors entirely different from the real paper
  - Title describes a fundamentally different topic from the real paper
  - Year off by 5+ years from the real paper
  - You cannot find any evidence the paper exists

**Score scale (0-10), used to support the label:**
- 9-10: perfect or near-perfect match (exact_match)
- 5-8: noticeable but the paper is still recognisable (minor_hallucination)
- 0-4: different paper, fabricated, or unverifiable (major_hallucination)

**Output format (REQUIRED).** Respond with a single JSON object — no markdown fences, no commentary before or after — with exactly these four keys:

{
  "label": "exact_match" | "minor_hallucination" | "major_hallucination",
  "score": <number between 0 and 10>,
  "reasoning": "<one to three short sentences justifying the decision>",
  "found_evidence": <true | false>
}
"""


WEB_SEARCH_NUDGE = """\

**You have access to a web search tool. Use it.** For every citation you must actively search the web to verify the authors, year, title, and identifiers against the real paper. Do not rely solely on memory.

**If, after searching, you cannot find or verify the paper:** output `"label": "major_hallucination"`, `"score": 0`, and `"found_evidence": false`. Treat unverifiable citations as fabrications.
"""


# ---------------------------------------------------------------------------
# User prompt builders
# ---------------------------------------------------------------------------

_EXAMPLE_BLOCK = """\
Example {idx}:
Citation: {citation_text}
Decision:
{{
  "label": "{label}",
  "score": {score},
  "reasoning": "{reasoning}",
  "found_evidence": {found_evidence}
}}
"""


def _score_for_label(label: str) -> int:
    """Pick a representative score consistent with the rubric.

    Used only to render few-shot exemplars; not part of the model's
    own scoring at test time.
    """
    return {
        "exact_match": 10,
        "minor_hallucination": 6,
        "major_hallucination": 1,
    }.get(label, 1)


def _escape_for_json_string(s: str) -> str:
    """Escape *s* so it can be embedded inside a JSON string literal
    that is itself embedded in a prompt example.
    """
    return s.replace("\\", "\\\\").replace('"', '\\"').replace("\n", " ").strip()


def _render_exemplar(idx: int, ex: Exemplar) -> str:
    return _EXAMPLE_BLOCK.format(
        idx=idx,
        citation_text=ex.citation_text.strip(),
        label=ex.label,
        score=_score_for_label(ex.label),
        reasoning=_escape_for_json_string(ex.reasoning),
        found_evidence="true" if ex.label != "major_hallucination" else "false",
    )


def build_user_prompt(
    citation_text: str,
    exemplars: Optional[Iterable[Exemplar]] = None,
) -> str:
    """Build the user message for a single citation.

    Parameters
    ----------
    citation_text
        Raw citation text as it appears in the report.
    exemplars
        If given, the exemplars are rendered as a numbered list before
        the target citation (few-shot prompting).  ``None`` produces a
        zero-shot prompt.
    """
    citation_text = citation_text.strip()

    if not exemplars:
        return (
            "Classify the following citation according to the rubric.\n\n"
            f"Citation:\n{citation_text}\n\n"
            "Respond with the JSON object only."
        )

    blocks = [_render_exemplar(i + 1, ex) for i, ex in enumerate(exemplars)]
    examples_text = "\n".join(blocks)

    return (
        "Below are labelled examples that demonstrate how to apply the "
        "rubric. Study them, then classify the final citation in the same "
        "format.\n\n"
        "=== EXAMPLES ===\n\n"
        f"{examples_text}\n"
        "=== TARGET ===\n\n"
        f"Citation: {citation_text}\n\n"
        "Respond with the JSON object only."
    )


def build_system_prompt(*, web_search: bool) -> str:
    """Return the system prompt, optionally with the web-search nudge."""
    if web_search:
        return BASELINE_SYSTEM_PROMPT + WEB_SEARCH_NUDGE
    return BASELINE_SYSTEM_PROMPT
