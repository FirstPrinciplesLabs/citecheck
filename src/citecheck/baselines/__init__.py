"""LLM-only baselines for comparison against CiteCheck.

This subpackage provides a comparison harness that evaluates frontier
LLMs (GPT, Claude, Gemini) on the same citation-classification task as
CiteCheck's main detector, in two modes — with and without web search
— using zero-shot or few-shot prompting.  Results are reported
alongside the detector's headline numbers so the contribution of each
design element (structured retrieval, dual-LLM review, etc.) can be
isolated.

Public API
----------
- :func:`call_baseline_llm` — single LLM call returning rich metrics.
- :func:`build_system_prompt`, :func:`build_user_prompt` — prompt templates.
- :func:`parse_baseline_output` — JSON-or-fallback output parser.
- :func:`run_baseline_config` — end-to-end runner for one configuration.
- :func:`evaluate_baseline_predictions`, :func:`save_evaluation` —
  evaluation primitives reusable on saved predictions.
- :func:`build_tag`, :func:`load_test_citations` — runner-level helpers.
"""

from .clients import (
    BaselineCallResult,
    PRICING_USD_PER_MTOK,
    SEARCH_PRICE_USD,
    SUPPORTED_PROVIDERS,
    call_baseline_llm,
    compute_token_cost,
)
from .evaluate import (
    evaluate_baseline_predictions,
    per_collection_metrics,
    save_evaluation,
    summarize_costs,
)
from .parser import ParsedOutput, VALID_LABELS, parse_baseline_output
from .prompts import (
    BASELINE_SYSTEM_PROMPT,
    WEB_SEARCH_NUDGE,
    build_system_prompt,
    build_user_prompt,
)
from .runner import build_tag, load_test_citations, run_baseline_config


__all__ = (
    "BaselineCallResult",
    "PRICING_USD_PER_MTOK",
    "SEARCH_PRICE_USD",
    "SUPPORTED_PROVIDERS",
    "call_baseline_llm",
    "compute_token_cost",
    "evaluate_baseline_predictions",
    "per_collection_metrics",
    "save_evaluation",
    "summarize_costs",
    "ParsedOutput",
    "VALID_LABELS",
    "parse_baseline_output",
    "BASELINE_SYSTEM_PROMPT",
    "WEB_SEARCH_NUDGE",
    "build_system_prompt",
    "build_user_prompt",
    "build_tag",
    "load_test_citations",
    "run_baseline_config",
)
