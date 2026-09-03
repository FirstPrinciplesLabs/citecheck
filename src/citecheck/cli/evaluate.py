"""``citecheck-evaluate`` -- dev/test evaluation harness.

Runs the full pipeline: build (or load) splits, optionally append a
few-shot block, run the detector on dev and test, tune
``(minor, exact)`` thresholds on dev, and evaluate dev / test at both
default (paper) and tuned thresholds.  Writes every artifact under
``--save-dir``.

Configuration sources (highest priority first):

1. Command-line flags.
2. YAML file passed via ``--config``.
3. Package defaults from :mod:`citecheck.config` (which
   match the paper's ``evaluate.sh`` job).

Examples
--------
End-to-end with the paper-default config (Anthropic / Claude / web search)::

    citecheck-evaluate --save-dir results/claude

Override one parameter from CLI::

    citecheck-evaluate --save-dir results/openai --provider openai --model gpt-5.2

Use a YAML file for the verifier config::

    citecheck-evaluate --config configs/evaluate-claude.yaml --save-dir results/claude

Re-tune thresholds on cached predictions (no API cost)::

    citecheck-evaluate --save-dir results/claude --no-cache=False \
        --dev-predictions  results/claude/dev_predictions.json \
        --test-predictions results/claude/test_predictions.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config
from ..eval.harness import evaluate_dataset
from ..llm.client import SUPPORTED_PROVIDERS
from ._yaml_config import load_yaml_config, merge_config


# Set of keys that *evaluate.yaml* is allowed to set.  Each maps 1:1
# to a kwarg of :func:`citecheck.eval.harness.evaluate_dataset`.
_VALID_YAML_KEYS = {
    "save_dir",
    "dataset_path",
    "seed",
    "splits_path",
    "default_exact_threshold",
    "default_minor_threshold",
    "dev_predictions_path",
    "test_predictions_path",
    "use_cache",
    "verbose",
    "detailed",
    "fewshot",
    # detector
    "try_arxiv",
    "try_web_search",
    "web_search_model",
    "accept_best_web_search",
    "api_timeout",
    "inter_api_delay",
    "llm_provider",
    "llm_model",
    "llm_temperature",
    "review_enabled",
    "review_model",
    "review_sim_threshold",
    "llm_parse_enabled",
    "llm_parse_model",
    "monitor",
}


# ---------------------------------------------------------------------------
# Argparse builder
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="citecheck-evaluate",
        description=(
            "Run the dev/test evaluation harness with threshold tuning. "
            "All defaults match the paper's evaluate.sh job."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # ── I/O ───────────────────────────────────────────────────────────
    io = parser.add_argument_group("I/O")
    io.add_argument(
        "--config", type=Path, default=None,
        help="Path to a YAML config file (CLI flags override YAML values).",
    )
    io.add_argument(
        "--save-dir", type=Path, default=None,
        help="Directory to write all outputs (created if missing).",
    )
    io.add_argument(
        "--dataset-path", type=Path, default=None,
        help=(
            "Override path to corruption_metadata.json "
            "(default: bundled dataset)."
        ),
    )
    io.add_argument(
        "--splits-path", type=Path, default=None,
        help=(
            "Path to a pre-computed splits.json (overrides --seed). "
            "Use this to share splits across the detector and baselines."
        ),
    )
    io.add_argument(
        "--dev-predictions", type=Path, default=None,
        dest="dev_predictions_path",
        help="Cached dev predictions.json (skip dev detection).",
    )
    io.add_argument(
        "--test-predictions", type=Path, default=None,
        dest="test_predictions_path",
        help="Cached test predictions.json (skip test detection).",
    )

    # ── Splits / mode ─────────────────────────────────────────────────
    splits = parser.add_argument_group("Splits / mode")
    splits.add_argument(
        "--seed", type=int, default=None,
        help=f"RNG seed for splits (default: {config.DEFAULT_SEED}).",
    )
    splits.add_argument(
        "--fewshot", action="store_true", default=None,
        help=(
            "Append the dev exemplars to the verifier prompt and remove "
            "them from the dev tuning set."
        ),
    )
    splits.add_argument(
        "--no-cache", action="store_const", const=False, default=None,
        dest="use_cache",
        help="Force re-running detection even if cached files exist.",
    )

    # ── Default thresholds ────────────────────────────────────────────
    thr = parser.add_argument_group("Default thresholds (pre-tuning)")
    thr.add_argument(
        "--exact-threshold", type=float, default=None,
        dest="default_exact_threshold",
        help=(
            "Score >= this -> exact_match "
            f"(default: {config.EXACT_THRESHOLD})."
        ),
    )
    thr.add_argument(
        "--minor-threshold", type=float, default=None,
        dest="default_minor_threshold",
        help=(
            "Score >= this -> minor_hallucination "
            f"(default: {config.MINOR_THRESHOLD})."
        ),
    )

    # ── LLM provider ──────────────────────────────────────────────────
    llm = parser.add_argument_group("LLM provider")
    llm.add_argument(
        "--provider", choices=SUPPORTED_PROVIDERS, default=None,
        dest="llm_provider",
        help=f"Verifier provider (default: {config.LLM_PROVIDER}).",
    )
    llm.add_argument(
        "--model", default=None, dest="llm_model",
        help=f"Verifier model name (default: {config.LLM_MODEL}).",
    )
    llm.add_argument(
        "--temperature", type=float, default=None, dest="llm_temperature",
        help=f"Sampling temperature (default: {config.LLM_TEMPERATURE}).",
    )

    # ── Cascade ───────────────────────────────────────────────────────
    cas = parser.add_argument_group("Cascade")
    cas.add_argument(
        "--no-arxiv", dest="try_arxiv", action="store_const", const=False, default=None,
        help="Disable the arXiv direct-ID lookup stage.",
    )
    cas.add_argument(
        "--no-web-search", dest="try_web_search", action="store_const", const=False, default=None,
        help="Disable the LLM web-search fallback stage.",
    )
    cas.add_argument(
        "--web-search-model", default=None,
        help=(
            "Model for web-search stage "
            f"(default: {config.WEB_SEARCH_MODEL})."
        ),
    )
    cas.add_argument(
        "--no-accept-best-web-search",
        dest="accept_best_web_search", action="store_const", const=False, default=None,
        help="Reject web-search candidates below similarity threshold.",
    )

    # ── Reviewer ──────────────────────────────────────────────────────
    rev = parser.add_argument_group("Reviewer")
    rev.add_argument(
        "--no-review", dest="review_enabled", action="store_const", const=False, default=None,
        help="Disable the second-pass reviewer LLM.",
    )
    rev.add_argument(
        "--review-model", default=None,
        help="Model for the reviewer LLM (default: same as --model).",
    )
    rev.add_argument(
        "--review-sim-threshold", type=float, default=None,
        help=(
            "Title similarity (0-100) below which the reviewer is "
            f"triggered (default: {config.REVIEW_SIM_THRESHOLD})."
        ),
    )

    # ── Citation parser ───────────────────────────────────────────────
    par = parser.add_argument_group("Citation parser")
    par.add_argument(
        "--no-llm-parse", dest="llm_parse_enabled",
        action="store_const", const=False, default=None,
        help="Disable LLM citation-parser fallback.",
    )
    par.add_argument(
        "--llm-parse-model", default=None,
        help="Model for the LLM citation parser (default: same as --model).",
    )

    # ── Verbosity ─────────────────────────────────────────────────────
    verbose = parser.add_argument_group("Verbosity")
    verbose.add_argument(
        "--quiet", dest="verbose", action="store_const", const=False, default=None,
        help="Suppress all progress output.",
    )
    verbose.add_argument(
        "--detailed", action="store_true", default=None,
        help="Print per-citation details during detection.",
    )
    verbose.add_argument(
        "--monitor", action="store_true", default=None,
        help="Record per-citation wall-clock / CPU / memory usage.",
    )

    return parser


# ---------------------------------------------------------------------------
# Default-value plumbing
# ---------------------------------------------------------------------------

def _harness_defaults() -> Dict[str, Any]:
    """Default kwargs for :func:`evaluate_dataset` derived from
    :mod:`citecheck.config`."""
    return {
        "save_dir": None,
        "dataset_path": None,
        "seed": config.DEFAULT_SEED,
        "splits_path": None,
        "default_exact_threshold": config.EXACT_THRESHOLD,
        "default_minor_threshold": config.MINOR_THRESHOLD,
        "dev_predictions_path": None,
        "test_predictions_path": None,
        "use_cache": True,
        "verbose": True,
        "detailed": False,
        "fewshot": False,
        # detector
        "try_arxiv": config.TRY_ARXIV,
        "try_web_search": config.TRY_WEB_SEARCH,
        "web_search_model": config.WEB_SEARCH_MODEL,
        "accept_best_web_search": config.ACCEPT_BEST_WEB_SEARCH,
        "api_timeout": config.API_TIMEOUT,
        "inter_api_delay": config.INTER_API_DELAY,
        "llm_provider": config.LLM_PROVIDER,
        "llm_model": None,  # falls through to config.LLM_MODEL inside the chain
        "llm_temperature": None,
        "review_enabled": config.REVIEW_ENABLED,
        "review_model": config.REVIEW_MODEL,
        "review_sim_threshold": config.REVIEW_SIM_THRESHOLD,
        "llm_parse_enabled": config.LLM_PARSE_ENABLED,
        "llm_parse_model": config.LLM_PARSE_MODEL,
        "monitor": False,
    }


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    yaml_cfg = load_yaml_config(args.config)

    cli_overrides: Dict[str, Any] = {
        "save_dir":                 args.save_dir,
        "dataset_path":             args.dataset_path,
        "seed":                     args.seed,
        "splits_path":              args.splits_path,
        "default_exact_threshold":  args.default_exact_threshold,
        "default_minor_threshold":  args.default_minor_threshold,
        "dev_predictions_path":     args.dev_predictions_path,
        "test_predictions_path":    args.test_predictions_path,
        "use_cache":                args.use_cache,
        "verbose":                  args.verbose,
        "detailed":                 args.detailed,
        "fewshot":                  args.fewshot,
        "try_arxiv":                args.try_arxiv,
        "try_web_search":           args.try_web_search,
        "web_search_model":         args.web_search_model,
        "accept_best_web_search":   args.accept_best_web_search,
        "llm_provider":             args.llm_provider,
        "llm_model":                args.llm_model,
        "llm_temperature":          args.llm_temperature,
        "review_enabled":           args.review_enabled,
        "review_model":             args.review_model,
        "review_sim_threshold":     args.review_sim_threshold,
        "llm_parse_enabled":        args.llm_parse_enabled,
        "llm_parse_model":          args.llm_parse_model,
        "monitor":                  args.monitor,
    }

    merged = merge_config(
        defaults=_harness_defaults(),
        yaml_cfg=yaml_cfg,
        cli_overrides=cli_overrides,
        valid_keys=_VALID_YAML_KEYS,
    )

    if merged.get("save_dir") is None:
        parser.error(
            "--save-dir is required (set it on the CLI or in the YAML config)."
        )

    merged["save_dir"] = Path(merged["save_dir"])
    for k in ("dataset_path", "splits_path",
              "dev_predictions_path", "test_predictions_path"):
        if merged.get(k) is not None:
            merged[k] = Path(merged[k])

    evaluate_dataset(**merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
