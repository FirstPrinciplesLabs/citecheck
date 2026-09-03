"""Robust parser for baseline LLM outputs.

The baseline prompt asks the model to return strict JSON with four
keys (``label``, ``score``, ``reasoning``, ``found_evidence``).  In
practice LLMs sometimes:

- wrap the JSON in markdown fences (```` ```json ... ``` ````)
- prepend explanatory text ("Here is the JSON: { ... }")
- emit a JSON-ish object with stray trailing commas or single quotes
- emit only a label keyword in free text ("major_hallucination")
- say "I cannot find this paper" with no JSON at all

This module tries each of these recovery paths in order.  On any hard
failure it falls back to the **conservative** answer mandated by the
project spec: ``major_hallucination`` with score 0 (treat unverifiable
or unparsable outputs as fabrications).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Optional


__all__ = (
    "ParsedOutput",
    "VALID_LABELS",
    "parse_baseline_output",
)


VALID_LABELS = (
    "exact_match",
    "minor_hallucination",
    "major_hallucination",
)

_NOT_FOUND_PHRASES = (
    "could not find",
    "couldn't find",
    "cannot find",
    "can not find",
    "unable to find",
    "no evidence",
    "does not exist",
    "doesn't exist",
    "not a real paper",
    "no such paper",
    "no record",
    "fabricated",
)


@dataclass
class ParsedOutput:
    """Normalised result extracted from a baseline LLM response."""

    label: str            # one of VALID_LABELS
    score: float          # 0-10
    reasoning: str
    found_evidence: bool
    parse_failure: bool   # True if any fallback path was used
    raw_text: str         # original LLM output, for debugging


# ---------------------------------------------------------------------------
# JSON extraction
# ---------------------------------------------------------------------------

_FENCED_JSON = re.compile(
    r"```(?:json)?\s*(\{.*?\})\s*```",
    re.DOTALL | re.IGNORECASE,
)

_BARE_JSON = re.compile(r"\{.*\}", re.DOTALL)


def _try_json_load(text: str) -> Optional[dict]:
    text = text.strip()
    if not text:
        return None
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except json.JSONDecodeError:
        return None


def _extract_json(raw: str) -> Optional[dict]:
    """Return the first dict-shaped JSON object recoverable from *raw*."""
    obj = _try_json_load(raw)
    if obj is not None:
        return obj

    m = _FENCED_JSON.search(raw)
    if m and (obj := _try_json_load(m.group(1))) is not None:
        return obj

    m = _BARE_JSON.search(raw)
    if m and (obj := _try_json_load(m.group(0))) is not None:
        return obj

    return None


# ---------------------------------------------------------------------------
# Field normalisation
# ---------------------------------------------------------------------------

def _coerce_label(value: Any, *, score: Optional[float] = None) -> Optional[str]:
    """Try to normalise a raw label value into one of :data:`VALID_LABELS`."""
    if isinstance(value, str):
        v = value.strip().lower().replace(" ", "_").replace("-", "_")
        if v in VALID_LABELS:
            return v
        if "exact" in v or v in ("correct", "valid", "match"):
            return "exact_match"
        if "minor" in v:
            return "minor_hallucination"
        if "major" in v or "fabric" in v or "halluc" in v:
            return "major_hallucination"

    if score is not None:
        if score >= 7:
            return "exact_match"
        if score >= 4:
            return "minor_hallucination"
        return "major_hallucination"

    return None


def _coerce_score(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return max(0.0, min(10.0, float(value)))
    if isinstance(value, str):
        m = re.search(r"-?\d+(?:\.\d+)?", value)
        if m:
            try:
                return max(0.0, min(10.0, float(m.group(0))))
            except ValueError:
                return None
    return None


def _coerce_bool(value: Any, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in ("true", "yes", "1"):
            return True
        if v in ("false", "no", "0"):
            return False
    return default


# ---------------------------------------------------------------------------
# Free-text fallback
# ---------------------------------------------------------------------------

def _looks_like_not_found(raw: str) -> bool:
    low = raw.lower()
    return any(p in low for p in _NOT_FOUND_PHRASES)


def _label_from_freetext(raw: str) -> Optional[str]:
    """Last-ditch label extraction by keyword matching."""
    low = raw.lower()
    for lbl in VALID_LABELS:
        if lbl in low:
            return lbl
    if _looks_like_not_found(low):
        return "major_hallucination"
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def parse_baseline_output(raw_text: str) -> ParsedOutput:
    """Extract a normalised classification from a raw LLM response.

    Falls back to ``major_hallucination`` (score 0,
    ``parse_failure=True``) if the response cannot be interpreted at
    all — consistent with the project rule that unverifiable citations
    are treated as fabrications.
    """
    raw_text = raw_text or ""

    obj = _extract_json(raw_text)
    if obj is not None:
        score = _coerce_score(obj.get("score"))
        label = _coerce_label(obj.get("label"), score=score)
        if label is None:
            label = _label_from_freetext(raw_text)

        if label is not None:
            if score is None:
                score = {
                    "exact_match": 9.5,
                    "minor_hallucination": 6.0,
                    "major_hallucination": 1.0,
                }[label]

            reasoning = obj.get("reasoning")
            if not isinstance(reasoning, str) or not reasoning.strip():
                reasoning = "(no reasoning provided)"

            found_default = label != "major_hallucination"
            found_evidence = _coerce_bool(
                obj.get("found_evidence"), default=found_default,
            )

            return ParsedOutput(
                label=label,
                score=float(score),
                reasoning=reasoning.strip(),
                found_evidence=found_evidence,
                parse_failure=False,
                raw_text=raw_text,
            )

    free_label = _label_from_freetext(raw_text)
    if free_label is not None:
        score = {
            "exact_match": 9.0,
            "minor_hallucination": 6.0,
            "major_hallucination": 0.0,
        }[free_label]
        return ParsedOutput(
            label=free_label,
            score=score,
            reasoning=raw_text.strip()[:500] or "(label inferred from free text)",
            found_evidence=free_label != "major_hallucination",
            parse_failure=True,
            raw_text=raw_text,
        )

    return ParsedOutput(
        label="major_hallucination",
        score=0.0,
        reasoning="(parse failure — defaulted to major_hallucination)",
        found_evidence=False,
        parse_failure=True,
        raw_text=raw_text,
    )
