"""``citecheck-compare`` -- run a matrix of LLM-only baselines and aggregate.

Reads a YAML *comparison matrix* describing several baseline runs
(provider × model × prompting × web-search), executes each through
:func:`citecheck.baselines.runner.run_baseline_config`, and writes:

- ``<save_dir>/<run_name>/...``    per-run baseline artefacts
- ``<save_dir>/comparison.json``   aggregated metrics + cost per run
- ``<save_dir>/comparison_table.txt`` human-readable comparison table

If ``include_main`` is set in the YAML, the comparison table is
augmented with a row reading from a previous ``citecheck-evaluate``
output directory (``test_evaluation.json``) so the main detector's
headline numbers show up alongside the LLM-only baselines.

YAML schema
-----------
.. code-block:: yaml

    save_dir: results/compare
    dataset_path: null         # optional, defaults to bundled dataset
    splits_path: null          # optional, share with citecheck-evaluate
    seed: 42

    # Inherited by every run unless overridden in `runs:`.
    shared:
      temperature: 0.0
      max_output_tokens: 1024
      web_search_max_uses: 5
      max_retries: 3
      max_citations: null
      save_every: 25

    runs:
      - name: gpt54_zero_nows
        provider: openai
        model: gpt-5.4
        prompting: zeroshot
        web_search: false
        reasoning_effort: medium
      - name: claude_few_ws
        provider: anthropic
        model: claude-sonnet-4-6
        prompting: fewshot
        web_search: true

    include_main:
      enabled: true
      results_path: results/claude        # citecheck-evaluate save_dir
      tag: CiteCheck (Claude)             # row label in the comparison

CLI
---
The most useful flags::

    citecheck-compare --config configs/baseline-matrix.yaml
    citecheck-compare --config matrix.yaml --save-dir results/compare2 \
        --max-citations 8                # smoke-test all rows
    citecheck-compare --config matrix.yaml --skip-existing  # resume
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence

from .. import config
from ..baselines.clients import SUPPORTED_PROVIDERS
from ..baselines.runner import build_tag, run_baseline_config
from ._yaml_config import load_yaml_config


__all__ = ("main",)


# ---------------------------------------------------------------------------
# YAML schema validation
# ---------------------------------------------------------------------------

_TOP_LEVEL_KEYS = {
    "save_dir",
    "dataset_path",
    "splits_path",
    "seed",
    "shared",
    "runs",
    "include_main",
}

_RUN_KEYS = {
    "name",
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

_INCLUDE_MAIN_KEYS = {"enabled", "results_path", "tag"}


@dataclass
class RunSpec:
    """One row in the comparison matrix."""

    name: str
    provider: str
    model: str
    prompting: str
    web_search: bool
    extra: Dict[str, Any]


def _validate_top(raw: Mapping[str, Any]) -> None:
    unknown = set(raw) - _TOP_LEVEL_KEYS
    if unknown:
        raise ValueError(
            f"Unknown top-level keys in YAML: {sorted(unknown)}. "
            f"Valid keys: {sorted(_TOP_LEVEL_KEYS)}"
        )


def _coerce_runs(
    raw_runs: Sequence[Mapping[str, Any]],
    shared: Mapping[str, Any],
) -> List[RunSpec]:
    if not raw_runs:
        raise ValueError("YAML must declare at least one run under `runs:`.")

    seen_names: set[str] = set()
    out: List[RunSpec] = []
    for i, item in enumerate(raw_runs):
        if not isinstance(item, Mapping):
            raise ValueError(f"`runs[{i}]` must be a mapping, got {type(item).__name__}")
        unknown = set(item) - _RUN_KEYS
        if unknown:
            raise ValueError(
                f"`runs[{i}]` has unknown keys: {sorted(unknown)}. "
                f"Valid keys: {sorted(_RUN_KEYS)}"
            )

        merged: Dict[str, Any] = {**shared, **{k: v for k, v in item.items() if v is not None}}
        for required in ("provider", "model", "prompting"):
            if not merged.get(required):
                raise ValueError(
                    f"`runs[{i}]` is missing required key {required!r} "
                    f"(can be set in `shared:` instead)."
                )

        provider = str(merged["provider"]).lower()
        if provider not in SUPPORTED_PROVIDERS:
            raise ValueError(
                f"`runs[{i}].provider` must be one of {SUPPORTED_PROVIDERS}, "
                f"got {provider!r}"
            )

        prompting = str(merged["prompting"]).lower()
        if prompting not in {"zeroshot", "fewshot"}:
            raise ValueError(
                f"`runs[{i}].prompting` must be 'zeroshot' or 'fewshot', "
                f"got {prompting!r}"
            )

        web_search = bool(merged.get("web_search", False))

        name = str(merged.get("name") or build_tag(
            provider, str(merged["model"]), prompting, web_search,
        ))
        if name in seen_names:
            raise ValueError(f"Duplicate run name {name!r} in `runs:`")
        seen_names.add(name)

        extra = {
            k: merged[k] for k in (
                "temperature", "reasoning_effort", "max_output_tokens",
                "web_search_max_uses", "max_retries",
                "max_citations", "save_every", "show_progress",
            ) if k in merged
        }

        out.append(RunSpec(
            name=name,
            provider=provider,
            model=str(merged["model"]),
            prompting=prompting,
            web_search=web_search,
            extra=extra,
        ))
    return out


def _validate_include_main(raw: Mapping[str, Any]) -> Dict[str, Any]:
    unknown = set(raw) - _INCLUDE_MAIN_KEYS
    if unknown:
        raise ValueError(
            f"Unknown keys in `include_main`: {sorted(unknown)}. "
            f"Valid keys: {sorted(_INCLUDE_MAIN_KEYS)}"
        )
    return {
        "enabled": bool(raw.get("enabled", False)),
        "results_path": raw.get("results_path"),
        "tag": str(raw.get("tag") or "CiteCheck (main)"),
    }


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _row_from_baseline(name: str, evaluation: Mapping[str, Any], run: RunSpec) -> Dict[str, Any]:
    gm = evaluation.get("global_metrics") or {}
    cs = evaluation.get("cost_summary") or {}
    return {
        "name": name,
        "kind": "baseline",
        "provider": run.provider,
        "model": run.model,
        "prompting": run.prompting,
        "web_search": run.web_search,
        "n": int(gm.get("total", 0)),
        "accuracy": float(gm.get("accuracy", 0.0)),
        "f1_exact": float(gm.get("exact_match", {}).get("f1", 0.0)),
        "f1_minor": float(gm.get("minor_hallucination", {}).get("f1", 0.0)),
        "f1_major": float(gm.get("major_hallucination", {}).get("f1", 0.0)),
        "total_cost_usd": float(cs.get("total_cost_usd", 0.0)),
        "mean_cost_per_citation_usd": float(cs.get("mean_cost_per_citation_usd", 0.0)),
        "mean_latency_s": float(cs.get("mean_latency_s", 0.0)),
        "p95_latency_s": float(cs.get("p95_latency_s", 0.0)),
        "total_search_calls": int(cs.get("total_search_calls", 0)),
        "parse_failures": int(cs.get("parse_failures", 0)),
        "call_errors": int(cs.get("call_errors", 0)),
    }


def _row_from_main(tag: str, results_path: Path) -> Optional[Dict[str, Any]]:
    """Build a comparison row from a ``citecheck-evaluate`` output dir.

    Reads ``test_evaluation.json`` and uses the *tuned* test-set metrics
    (the headline numbers reported in the paper).  Returns ``None`` (with
    a printed warning) if the file is missing or malformed so an absent
    main-detector run doesn't fail the whole compare.
    """
    eval_path = Path(results_path) / "test_evaluation.json"
    if not eval_path.exists():
        print(f"[warn] include_main: {eval_path} not found — skipping row")
        return None

    try:
        payload = json.loads(eval_path.read_text(encoding="utf-8"))
    except Exception as exc:                                  # noqa: BLE001
        print(f"[warn] include_main: could not parse {eval_path}: {exc}")
        return None

    tuned = payload.get("tuned") or payload.get("default") or {}
    gm = tuned.get("global_metrics") or {}
    if not gm:
        print(f"[warn] include_main: no tuned global_metrics in {eval_path}")
        return None

    return {
        "name": tag,
        "kind": "main",
        "provider": "—",
        "model": "—",
        "prompting": "—",
        "web_search": "—",
        "n": int(gm.get("total", 0)),
        "accuracy": float(gm.get("accuracy", 0.0)),
        "f1_exact": float(gm.get("exact_match", {}).get("f1", 0.0)),
        "f1_minor": float(gm.get("minor_hallucination", {}).get("f1", 0.0)),
        "f1_major": float(gm.get("major_hallucination", {}).get("f1", 0.0)),
        "total_cost_usd": None,
        "mean_cost_per_citation_usd": None,
        "mean_latency_s": None,
        "p95_latency_s": None,
        "total_search_calls": None,
        "parse_failures": None,
        "call_errors": None,
        "results_path": str(results_path),
    }


# ---------------------------------------------------------------------------
# Comparison table rendering
# ---------------------------------------------------------------------------

_TABLE_COLUMNS: Sequence[tuple[str, str, str]] = (
    ("name",                       "Run",            "{:<28s}"),
    ("provider",                   "Provider",       "{:<10s}"),
    ("model",                      "Model",          "{:<22s}"),
    ("prompting",                  "Prompt",         "{:<8s}"),
    ("web_search",                 "WS",             "{!s:<5s}"),
    ("n",                          "N",              "{:>4}"),
    ("accuracy",                   "Acc",            "{:>6.3f}"),
    ("f1_exact",                   "F1 ex",          "{:>6.3f}"),
    ("f1_minor",                   "F1 mi",          "{:>6.3f}"),
    ("f1_major",                   "F1 ma",          "{:>6.3f}"),
    ("total_cost_usd",             "Cost $",         "{:>8.3f}"),
    ("mean_cost_per_citation_usd", "$/cit",          "{:>8.5f}"),
    ("mean_latency_s",             "lat μ",          "{:>6.2f}"),
    ("p95_latency_s",              "lat p95",        "{:>7.2f}"),
)


def _format_cell(value: Any, fmt: str) -> str:
    if value is None or value == "—":
        return "—"
    try:
        return fmt.format(value)
    except Exception:
        return str(value)


def render_comparison_table(rows: Sequence[Mapping[str, Any]]) -> str:
    if not rows:
        return "(no rows)"

    headers = [h for _, h, _ in _TABLE_COLUMNS]
    fmts = [f for _, _, f in _TABLE_COLUMNS]

    formatted: List[List[str]] = []
    for row in rows:
        formatted.append([
            _format_cell(row.get(key), fmt)
            for (key, _, fmt) in _TABLE_COLUMNS
        ])

    col_widths = [max(len(h), max(len(r[i]) for r in formatted))
                  for i, h in enumerate(headers)]

    def _join(cells: Sequence[str]) -> str:
        return "  ".join(c.ljust(w) for c, w in zip(cells, col_widths))

    lines = [_join(headers), _join(["-" * w for w in col_widths])]
    for cells in formatted:
        lines.append(_join(cells))
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def _run_matrix(
    *,
    save_dir: Path,
    runs: Sequence[RunSpec],
    dataset_path: Optional[Path],
    splits_path: Optional[Path],
    seed: int,
    skip_existing: bool,
    cli_max_citations: Optional[int],
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for run in runs:
        run_dir = save_dir / run.name
        run_dir.mkdir(parents=True, exist_ok=True)

        if skip_existing and (run_dir / "evaluation_results.json").exists():
            print(f"\n[skip] {run.name}: evaluation_results.json already exists")
            try:
                payload = json.loads(
                    (run_dir / "evaluation_results.json").read_text(encoding="utf-8")
                )
                rows.append(_row_from_baseline(run.name, payload, run))
            except Exception as exc:                          # noqa: BLE001
                print(f"[warn] could not reload {run_dir}: {exc}")
            continue

        kwargs: Dict[str, Any] = {
            "provider": run.provider,
            "model": run.model,
            "prompting": run.prompting,
            "web_search": run.web_search,
            "save_dir": run_dir,
            "dataset_path": dataset_path,
            "splits_path": splits_path,
            "seed": seed,
        }
        kwargs.update(run.extra)
        if cli_max_citations is not None:
            kwargs["max_citations"] = cli_max_citations

        print(f"\n{'#' * 72}")
        print(f"# Running {run.name}  ({run.provider} / {run.model} / "
              f"{run.prompting} / {'WS' if run.web_search else 'noWS'})")
        print(f"{'#' * 72}")
        evaluation = run_baseline_config(**kwargs)
        rows.append(_row_from_baseline(run.name, evaluation, run))
    return rows


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="citecheck-compare",
        description=(
            "Run a matrix of LLM-only baselines from a YAML config and "
            "aggregate their metrics + cost into a comparison table."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--config", type=Path, required=True,
        help="Path to the YAML matrix config (required).",
    )
    parser.add_argument(
        "--save-dir", type=Path, default=None,
        help="Override `save_dir` in the YAML.",
    )
    parser.add_argument(
        "--dataset-path", type=Path, default=None,
        help="Override `dataset_path` in the YAML.",
    )
    parser.add_argument(
        "--splits-path", type=Path, default=None,
        help="Override `splits_path` in the YAML.",
    )
    parser.add_argument(
        "--seed", type=int, default=None,
        help=f"Override `seed` (default: {config.DEFAULT_SEED}).",
    )
    parser.add_argument(
        "--max-citations", type=int, default=None,
        help="Force-cap the test-set size on every row (smoke testing).",
    )
    parser.add_argument(
        "--skip-existing", action="store_true",
        help="Skip rows whose evaluation_results.json already exists.",
    )
    return parser


# ---------------------------------------------------------------------------
# main()
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    raw = load_yaml_config(args.config)
    _validate_top(raw)

    save_dir = Path(args.save_dir or raw.get("save_dir") or "")
    if not str(save_dir):
        parser.error("`save_dir` must be set on the CLI or in the YAML.")
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = (
        Path(args.dataset_path) if args.dataset_path
        else (Path(raw["dataset_path"]) if raw.get("dataset_path") else None)
    )
    splits_path = (
        Path(args.splits_path) if args.splits_path
        else (Path(raw["splits_path"]) if raw.get("splits_path") else None)
    )
    seed = int(args.seed if args.seed is not None else raw.get("seed", config.DEFAULT_SEED))

    shared = dict(raw.get("shared") or {})
    runs = _coerce_runs(list(raw.get("runs") or []), shared)

    include_main = _validate_include_main(dict(raw.get("include_main") or {}))

    started = time.strftime("%Y-%m-%dT%H:%M:%S")
    print(f"\n[citecheck-compare] save_dir={save_dir}  runs={len(runs)}")

    baseline_rows = _run_matrix(
        save_dir=save_dir,
        runs=runs,
        dataset_path=dataset_path,
        splits_path=splits_path,
        seed=seed,
        skip_existing=args.skip_existing,
        cli_max_citations=args.max_citations,
    )

    rows: List[Dict[str, Any]] = list(baseline_rows)
    if include_main["enabled"]:
        rp = include_main["results_path"]
        if rp is None:
            print("[warn] include_main.enabled=true but no `results_path` set")
        else:
            main_row = _row_from_main(include_main["tag"], Path(rp))
            if main_row is not None:
                rows.insert(0, main_row)

    finished = time.strftime("%Y-%m-%dT%H:%M:%S")

    payload = {
        "started_at": started,
        "finished_at": finished,
        "save_dir": str(save_dir),
        "n_runs": len(runs),
        "include_main": include_main,
        "rows": rows,
    }
    (save_dir / "comparison.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8",
    )

    table = render_comparison_table(rows)
    (save_dir / "comparison_table.txt").write_text(
        f"# citecheck-compare ({started} → {finished})\n"
        f"# save_dir: {save_dir}\n\n{table}\n",
        encoding="utf-8",
    )

    print("\n" + "=" * 72)
    print("COMPARISON TABLE")
    print("=" * 72)
    print(table)
    print(f"\nWrote {save_dir / 'comparison.json'}")
    print(f"Wrote {save_dir / 'comparison_table.txt'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
