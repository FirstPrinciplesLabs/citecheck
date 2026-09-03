"""``citecheck-baseline`` -- run one LLM-only baseline configuration.

Single (provider × model × prompting × web-search) run over the test
split.  Persists predictions, cost / latency aggregates, and metrics
under ``--save-dir`` so that ``citecheck-compare`` can later join the
results with the main detector's headline numbers.

Configuration sources (highest priority first):

1. Command-line flags.
2. YAML file passed via ``--config``.
3. Sensible defaults (``temperature=0.0``, ``web_search=False``,
   ``prompting="zeroshot"``, etc.).

Examples
--------
Zero-shot, no-web-search GPT::

    citecheck-baseline \\
        --provider openai --model gpt-5.4 \\
        --prompting zeroshot \\
        --save-dir results/baselines/openai_gpt54_zero_nows

Few-shot Anthropic with web search, capped at 8 citations for a smoke
test (shares splits with a previous ``citecheck-evaluate`` run)::

    citecheck-baseline \\
        --provider anthropic --model claude-sonnet-4-6 \\
        --prompting fewshot --web-search \\
        --splits-path results/claude/splits.json \\
        --max-citations 8 \\
        --save-dir results/baselines/claude_few_ws

YAML-driven::

    citecheck-baseline --config configs/baseline-claude.yaml \\
        --save-dir results/baselines/claude_zero_nows
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import config
from ..baselines.clients import SUPPORTED_PROVIDERS
from ..baselines.runner import run_baseline_config
from ._yaml_config import load_yaml_config, merge_config


_VALID_YAML_KEYS = {
    "save_dir",
    "dataset_path",
    "splits_path",
    "seed",
    "provider",
    "model",
    "prompting",
    "web_search",
    "temperature",
    "reasoning_effort",
    "max_output_tokens",
    "web_search_max_uses",
    "max_retries",
    "max_citations",
    "save_every",
    "show_progress",
}


def _baseline_defaults() -> Dict[str, Any]:
    return {
        "save_dir": None,
        "dataset_path": None,
        "splits_path": None,
        "seed": config.DEFAULT_SEED,
        "provider": None,
        "model": None,
        "prompting": "zeroshot",
        "web_search": False,
        "temperature": 0.0,
        "reasoning_effort": None,
        "max_output_tokens": 1024,
        "web_search_max_uses": 5,
        "max_retries": 3,
        "max_citations": None,
        "save_every": 25,
        "show_progress": True,
    }


# ---------------------------------------------------------------------------
# Argparse builder
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="citecheck-baseline",
        description=(
            "Run one LLM-only baseline configuration over the test split "
            "and write predictions + metrics under --save-dir."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    io = parser.add_argument_group("I/O")
    io.add_argument(
        "--config", type=Path, default=None,
        help="Optional YAML config (CLI flags override YAML values).",
    )
    io.add_argument(
        "--save-dir", type=Path, default=None,
        help="Directory to write run outputs (created if missing).",
    )
    io.add_argument(
        "--dataset-path", type=Path, default=None,
        help="Override path to corruption_metadata.json.",
    )
    io.add_argument(
        "--splits-path", type=Path, default=None,
        help=(
            "Path to a pre-computed splits.json (overrides --seed). "
            "Use this to share splits with the main detector."
        ),
    )

    cfg = parser.add_argument_group("Run configuration")
    cfg.add_argument(
        "--provider", choices=SUPPORTED_PROVIDERS, default=None,
        help="LLM provider.",
    )
    cfg.add_argument(
        "--model", default=None,
        help="Provider-specific model identifier.",
    )
    cfg.add_argument(
        "--prompting", choices=["zeroshot", "fewshot"], default=None,
        help="Prompting mode (default: zeroshot).",
    )
    cfg.add_argument(
        "--web-search", dest="web_search",
        action="store_const", const=True, default=None,
        help="Enable the provider's native web-search tool.",
    )
    cfg.add_argument(
        "--no-web-search", dest="web_search",
        action="store_const", const=False, default=None,
        help="Force-disable the web-search tool.",
    )

    sp = parser.add_argument_group("Splits")
    sp.add_argument(
        "--seed", type=int, default=None,
        help=f"RNG seed for splits (default: {config.DEFAULT_SEED}).",
    )

    sm = parser.add_argument_group("Sampling")
    sm.add_argument(
        "--temperature", type=float, default=None,
        help="Sampling temperature (default: 0.0).",
    )
    sm.add_argument(
        "--reasoning-effort", default=None,
        choices=["none", "low", "medium", "high"],
        help="OpenAI gpt-5.x reasoning effort (ignored for other providers).",
    )
    sm.add_argument(
        "--max-output-tokens", type=int, default=None, dest="max_output_tokens",
        help="Cap on completion length (default: 1024).",
    )
    sm.add_argument(
        "--web-search-max-uses", type=int, default=None,
        dest="web_search_max_uses",
        help=(
            "Cap on tool-call count (Anthropic only; OpenAI/Gemini ignore). "
            "Default: 5."
        ),
    )
    sm.add_argument(
        "--max-retries", type=int, default=None, dest="max_retries",
        help="Number of attempts on transient errors (default: 3).",
    )

    misc = parser.add_argument_group("Run loop")
    misc.add_argument(
        "--max-citations", type=int, default=None, dest="max_citations",
        help="Cap test-set size for smoke tests (default: all).",
    )
    misc.add_argument(
        "--save-every", type=int, default=None, dest="save_every",
        help="Persist intermediate predictions every N citations (default: 25).",
    )
    misc.add_argument(
        "--quiet", dest="show_progress",
        action="store_const", const=False, default=None,
        help="Suppress the per-citation progress bar.",
    )

    return parser


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    yaml_cfg = load_yaml_config(args.config)

    cli_overrides: Dict[str, Any] = {
        "save_dir":            args.save_dir,
        "dataset_path":        args.dataset_path,
        "splits_path":         args.splits_path,
        "seed":                args.seed,
        "provider":            args.provider,
        "model":               args.model,
        "prompting":           args.prompting,
        "web_search":          args.web_search,
        "temperature":         args.temperature,
        "reasoning_effort":    args.reasoning_effort,
        "max_output_tokens":   args.max_output_tokens,
        "web_search_max_uses": args.web_search_max_uses,
        "max_retries":         args.max_retries,
        "max_citations":       args.max_citations,
        "save_every":          args.save_every,
        "show_progress":       args.show_progress,
    }

    merged = merge_config(
        defaults=_baseline_defaults(),
        yaml_cfg=yaml_cfg,
        cli_overrides=cli_overrides,
        valid_keys=_VALID_YAML_KEYS,
    )

    for required in ("save_dir", "provider", "model"):
        if merged.get(required) is None:
            parser.error(
                f"--{required.replace('_', '-')} is required "
                "(set it on the CLI or in the YAML config)."
            )

    merged["save_dir"] = Path(merged["save_dir"])
    for k in ("dataset_path", "splits_path"):
        if merged.get(k) is not None:
            merged[k] = Path(merged[k])

    run_baseline_config(**merged)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
