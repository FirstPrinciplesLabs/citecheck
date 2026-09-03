"""Run one (provider × model × prompting × web_search) baseline configuration.

A single invocation iterates over the test split's citations, calls
the chosen LLM once per citation, parses the output, evaluates against
ground truth, and writes everything under
``save_dir/<tag>/`` ::

    config.json              configuration of the run
    predictions.json         per-citation outputs + metrics
    cost_summary.json        aggregate cost / latency / search totals
    metrics_summary.txt      human-readable metrics report
    evaluation_results.json  full evaluation payload

The runner is intentionally single-config so that orchestrators (e.g.
``citecheck-compare`` or a SLURM array job) can parallelise across
configurations without any custom worker logic here.

Determinism
-----------
The dev/test split and few-shot pool are deterministic given the same
seed (default :data:`citecheck.config.DEFAULT_SEED`).
``temperature=0`` is used by default.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

from tqdm import tqdm

from .. import config
from ..eval.dataset import Dataset, load_dataset
from ..eval.ground_truth import build_gt_map
from ..eval.splits import (
    Exemplar,
    Splits,
    load_splits,
    make_fewshot_exemplars,
    make_splits,
    save_splits,
)
from .clients import (
    BaselineCallResult,
    SUPPORTED_PROVIDERS,
    call_baseline_llm,
)
from .evaluate import evaluate_baseline_predictions, save_evaluation
from .parser import ParsedOutput, parse_baseline_output
from .prompts import build_system_prompt, build_user_prompt


__all__ = (
    "build_tag",
    "load_test_citations",
    "run_baseline_config",
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def build_tag(
    provider: str, model: str, prompting: str, web_search: bool,
) -> str:
    """Produce a filesystem-safe identifier for one baseline configuration."""
    safe_model = model.replace("/", "_").replace(":", "_")
    ws = "ws" if web_search else "nows"
    return f"{provider}_{safe_model}_{prompting}_{ws}"


def _resolve_dataset_and_splits(
    *,
    dataset: Optional[Dataset],
    dataset_path: Optional[Path],
    splits: Optional[Splits],
    exemplars: Optional[Sequence[Exemplar]],
    splits_path: Optional[Path],
    seed: int,
    save_splits_to: Optional[Path],
) -> tuple[Dataset, Splits, List[Exemplar]]:
    if dataset is None:
        dataset = load_dataset(dataset_path) if dataset_path else load_dataset()

    if splits is None:
        if splits_path is not None and Path(splits_path).exists():
            print(f"Loading splits from {splits_path}")
            splits, ex_loaded = load_splits(Path(splits_path))
            return dataset, splits, list(ex_loaded)

        print(f"Computing splits (seed={seed})")
        splits = make_splits(dataset, seed=seed)
        ex_built = make_fewshot_exemplars(dataset, splits, seed=seed)
        if save_splits_to is not None:
            save_splits(splits, ex_built, Path(save_splits_to))
            print(f"Saved splits → {save_splits_to}")
        return dataset, splits, list(ex_built)

    return dataset, splits, list(exemplars or [])


def load_test_citations(
    dataset: Dataset, splits: Splits,
) -> List[Dict[str, Any]]:
    """Return one dict per test citation, ready to be classified.

    Each dict carries: ``collection_id``, ``collection_number``,
    ``topic``, ``subtopic``, ``citation_number``, ``citation_text``
    (the *corrupted* citation as it appears in the report), and
    ``ground_truth_raw`` (the dataset's raw label, pre-mapping).
    """
    test_set = set(splits.test_collections)
    out: List[Dict[str, Any]] = []
    for col in dataset.collections:
        if col.collection_number not in test_set:
            continue
        for c in col.citations:
            text = c.corrupted_citation or c.original_citation or ""
            if not text.strip():
                continue
            out.append({
                "collection_id": col.collection_id,
                "collection_number": col.collection_number,
                "topic": col.topic,
                "subtopic": col.subtopic,
                "citation_number": c.citation_number,
                "citation_text": text,
                "ground_truth_raw": c.ground_truth_label,
            })
    return out


def _result_to_record(
    item: Dict[str, Any],
    parsed: ParsedOutput,
    call: BaselineCallResult,
    *,
    provider: str,
    model: str,
    prompting: str,
    web_search: bool,
) -> Dict[str, Any]:
    return {
        "collection_id": item["collection_id"],
        "collection_number": item["collection_number"],
        "topic": item.get("topic"),
        "subtopic": item.get("subtopic"),
        "citation_number": item["citation_number"],
        "citation_text": item["citation_text"],
        "ground_truth_raw": item["ground_truth_raw"],
        "provider": provider,
        "model": model,
        "prompting": prompting,
        "web_search": web_search,
        "predicted_label": parsed.label,
        "score": parsed.score,
        "reasoning": parsed.reasoning,
        "found_evidence": parsed.found_evidence,
        "parse_failure": parsed.parse_failure,
        "raw_text": parsed.raw_text,
        "prompt_tokens": call.prompt_tokens,
        "completion_tokens": call.completion_tokens,
        "total_tokens": call.total_tokens,
        "latency_s": call.latency_s,
        "cost_usd": call.cost_usd,
        "attempts": call.attempts,
        "call_error": call.error,
        "web_search_calls": int(call.extra.get("web_search_calls", 0)),
        "token_cost_usd": call.extra.get("token_cost_usd"),
        "search_cost_usd": call.extra.get("search_cost_usd"),
    }


# ---------------------------------------------------------------------------
# Main run loop
# ---------------------------------------------------------------------------

def run_baseline_config(
    *,
    provider: str,
    model: str,
    prompting: str,
    web_search: bool,
    save_dir: Path,
    dataset: Optional[Dataset] = None,
    dataset_path: Optional[Path] = None,
    splits: Optional[Splits] = None,
    exemplars: Optional[Sequence[Exemplar]] = None,
    splits_path: Optional[Path] = None,
    seed: int = config.DEFAULT_SEED,
    temperature: Optional[float] = 0.0,
    reasoning_effort: Optional[str] = None,
    max_output_tokens: int = 1024,
    web_search_max_uses: int = 5,
    max_retries: int = 3,
    max_citations: Optional[int] = None,
    save_every: int = 25,
    show_progress: bool = True,
) -> Dict[str, Any]:
    """Run one baseline configuration end-to-end and persist artefacts.

    Parameters mirror the CLI flags of ``citecheck-baseline``; see the
    module docstring for the on-disk layout.

    Returns the in-memory evaluation payload (same shape as
    :func:`citecheck.baselines.evaluate.evaluate_baseline_predictions`).
    """
    if prompting not in {"zeroshot", "fewshot"}:
        raise ValueError(
            f"prompting must be 'zeroshot' or 'fewshot', got {prompting!r}"
        )
    if provider not in SUPPORTED_PROVIDERS:
        raise ValueError(
            f"Unsupported provider {provider!r}; choose from {SUPPORTED_PROVIDERS}"
        )

    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    dataset, splits, exemplar_list = _resolve_dataset_and_splits(
        dataset=dataset,
        dataset_path=dataset_path,
        splits=splits,
        exemplars=exemplars,
        splits_path=splits_path,
        seed=seed,
        save_splits_to=save_dir / "splits.json",
    )

    test_items = load_test_citations(dataset, splits)
    if max_citations is not None:
        test_items = test_items[:max_citations]

    system_prompt = build_system_prompt(web_search=web_search)
    pool: Optional[List[Exemplar]] = (
        list(exemplar_list) if prompting == "fewshot" else None
    )

    config_payload: Dict[str, Any] = {
        "provider": provider,
        "model": model,
        "prompting": prompting,
        "web_search": web_search,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": max_output_tokens,
        "web_search_max_uses": web_search_max_uses,
        "max_retries": max_retries,
        "n_test_citations": len(test_items),
        "n_test_collections": len(splits.test_collections),
        "n_exemplars": len(pool) if pool else 0,
        "splits_seed": splits.seed,
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    (save_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8",
    )

    predictions: List[Dict[str, Any]] = []
    desc = f"{provider}/{model} {prompting} {'WS' if web_search else 'noWS'}"
    iterator: Iterable[Dict[str, Any]]
    if show_progress:
        iterator = tqdm(test_items, desc=desc, total=len(test_items))
    else:
        iterator = test_items

    running_cost = 0.0
    for i, item in enumerate(iterator):
        user_prompt = build_user_prompt(item["citation_text"], exemplars=pool)

        call = call_baseline_llm(
            provider=provider,
            model=model,
            system=system_prompt,
            user=user_prompt,
            web_search=web_search,
            temperature=temperature,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            web_search_max_uses=web_search_max_uses if web_search else None,
            max_retries=max_retries,
        )
        parsed = parse_baseline_output(call.text)
        record = _result_to_record(
            item, parsed, call,
            provider=provider, model=model,
            prompting=prompting, web_search=web_search,
        )
        predictions.append(record)
        running_cost += call.cost_usd
        if show_progress and isinstance(iterator, tqdm):
            iterator.set_postfix({
                "cost$": f"{running_cost:.3f}",
                "err": int(bool(call.error)),
            })

        if save_every and (i + 1) % save_every == 0:
            (save_dir / "predictions.json").write_text(
                json.dumps(
                    {"config": config_payload, "predictions": predictions},
                    indent=2, default=str,
                ),
                encoding="utf-8",
            )

    config_payload["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    (save_dir / "config.json").write_text(
        json.dumps(config_payload, indent=2), encoding="utf-8",
    )

    gt_map = build_gt_map(dataset)
    tag = build_tag(provider, model, prompting, web_search)
    evaluation = evaluate_baseline_predictions(
        predictions, gt_map, tag=tag, verbose=True,
    )
    save_evaluation(predictions, evaluation, save_dir, tag=tag)

    print(f"\nResults written to {save_dir}")
    return evaluation
