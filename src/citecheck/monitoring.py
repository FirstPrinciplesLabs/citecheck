"""Resource-monitoring helpers.

Lightweight wall-clock + (optional) CPU/memory tracking used by the
detector and the eval harness.  ``psutil`` is an optional extra: when
unavailable, only wall-clock timing is captured and CPU/memory fields
default to zero.
"""

from __future__ import annotations

import time
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Any, ContextManager, Dict, List, Optional

try:
    import psutil

    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False


__all__ = (
    "ResourceSnapshot",
    "ResourceMonitor",
    "take_snapshot",
    "monitor_context",
    "aggregate_metrics",
    "aggregate_collection_metrics",
)


# ---------------------------------------------------------------------------
# Snapshots and per-block monitor
# ---------------------------------------------------------------------------

@dataclass
class ResourceSnapshot:
    """Point-in-time snapshot of process resource usage."""
    timestamp: float = 0.0
    cpu_user: float = 0.0
    cpu_system: float = 0.0
    memory_rss_bytes: int = 0
    memory_vms_bytes: int = 0


def take_snapshot() -> ResourceSnapshot:
    """Capture current process resource usage."""
    ts = time.perf_counter()
    if _HAS_PSUTIL:
        proc = psutil.Process()
        cpu = proc.cpu_times()
        mem = proc.memory_info()
        return ResourceSnapshot(
            timestamp=ts,
            cpu_user=cpu.user,
            cpu_system=cpu.system,
            memory_rss_bytes=mem.rss,
            memory_vms_bytes=mem.vms,
        )
    return ResourceSnapshot(timestamp=ts)


class ResourceMonitor:
    """Context manager that captures resource deltas over a code block.

    Example::

        with ResourceMonitor() as mon:
            heavy_work()
        print(mon.metrics)
    """

    def __init__(self) -> None:
        self._start: Optional[ResourceSnapshot] = None
        self._end: Optional[ResourceSnapshot] = None

    def __enter__(self) -> "ResourceMonitor":
        self._start = take_snapshot()
        return self

    def __exit__(self, *args: object) -> None:
        self._end = take_snapshot()

    @property
    def wall_time(self) -> float:
        """Wall-clock seconds elapsed."""
        if self._start and self._end:
            return self._end.timestamp - self._start.timestamp
        return 0.0

    @property
    def metrics(self) -> Dict[str, Any]:
        """Resource-usage deltas over the block."""
        if not self._start or not self._end:
            return {}
        s, e = self._start, self._end
        return {
            "wall_time_seconds": round(e.timestamp - s.timestamp, 4),
            "cpu_user_seconds": round(e.cpu_user - s.cpu_user, 4),
            "cpu_system_seconds": round(e.cpu_system - s.cpu_system, 4),
            "memory_rss_mb": round(e.memory_rss_bytes / (1024 * 1024), 2),
            "memory_rss_delta_mb": round(
                (e.memory_rss_bytes - s.memory_rss_bytes) / (1024 * 1024), 2
            ),
        }

    @property
    def has_resource_data(self) -> bool:
        """True when psutil is available."""
        return _HAS_PSUTIL


def monitor_context(enabled: bool) -> ContextManager:
    """Return a :class:`ResourceMonitor` when *enabled*, else a no-op context."""
    if enabled:
        return ResourceMonitor()
    return nullcontext()


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def _percentile(sorted_vals: List[float], pct: float) -> float:
    """Linearly interpolated percentile from a *pre-sorted* list."""
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    k = (n - 1) * pct / 100.0
    f = int(k)
    c = f + 1 if f + 1 < n else f
    d = k - f
    return sorted_vals[f] + d * (sorted_vals[c] - sorted_vals[f])


def aggregate_metrics(
    records: List[Dict[str, Any]],
    *,
    stage_keys: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """Compute aggregate performance statistics from per-item records.

    Parameters
    ----------
    records
        Each record should have a ``"perf"`` dict with at least
        ``wall_time_seconds``, plus the per-stage timing fields the
        detector emits (``api_search_seconds``, ``llm_inference_seconds``,
        ``api_timings``).
    stage_keys
        Names of arbitrary per-stage timing keys to aggregate.

    Returns
    -------
    dict
        Latency distribution, CPU totals, peak memory, plus per-API and
        per-stage breakdowns when present.
    """
    perfs = [r["perf"] for r in records if "perf" in r]
    if not perfs:
        return {}

    wall_times = [p["wall_time_seconds"] for p in perfs]
    wall_times_sorted = sorted(wall_times)
    n = len(wall_times_sorted)

    agg: Dict[str, Any] = {
        "item_count": n,
        "latency_mean_s": round(sum(wall_times) / n, 4),
        "latency_median_s": round(_percentile(wall_times_sorted, 50), 4),
        "latency_p95_s": round(_percentile(wall_times_sorted, 95), 4),
        "latency_min_s": round(wall_times_sorted[0], 4),
        "latency_max_s": round(wall_times_sorted[-1], 4),
        "latency_total_s": round(sum(wall_times), 2),
    }

    cpu_user = [p.get("cpu_user_seconds", 0) for p in perfs]
    cpu_sys = [p.get("cpu_system_seconds", 0) for p in perfs]
    if any(v > 0 for v in cpu_user):
        agg["cpu_user_total_s"] = round(sum(cpu_user), 4)
        agg["cpu_system_total_s"] = round(sum(cpu_sys), 4)

    rss_vals = [p.get("memory_rss_mb", 0) for p in perfs]
    if any(v > 0 for v in rss_vals):
        agg["memory_peak_rss_mb"] = round(max(rss_vals), 2)

    if stage_keys:
        stage_agg: Dict[str, Dict[str, Any]] = {}
        for key in stage_keys:
            vals = [p.get(key, 0) for p in perfs if key in p]
            if vals:
                total = sum(vals)
                stage_agg[key] = {
                    "count": len(vals),
                    "mean_s": round(total / len(vals), 4),
                    "total_s": round(total, 4),
                }
        if stage_agg:
            agg["per_stage_latency"] = stage_agg

    api_times = [p.get("api_search_seconds", 0) for p in perfs if "api_search_seconds" in p]
    llm_times = [p.get("llm_inference_seconds", 0) for p in perfs if "llm_inference_seconds" in p]
    if api_times:
        total_api = sum(api_times)
        total_llm = sum(llm_times)
        total_both = total_api + total_llm
        agg["api_search_mean_s"] = round(total_api / len(api_times), 4)
        agg["llm_inference_mean_s"] = (
            round(total_llm / len(llm_times), 4) if llm_times else 0
        )
        if total_both > 0:
            agg["api_search_pct"] = round(100 * total_api / total_both, 1)
            agg["llm_inference_pct"] = round(100 * total_llm / total_both, 1)

    api_names = ["crossref", "semantic_scholar", "openalex", "arxiv", "web_search"]
    per_api: Dict[str, List[float]] = {name: [] for name in api_names}
    for p in perfs:
        api_timings = p.get("api_timings", {})
        for name in api_names:
            if name in api_timings:
                per_api[name].append(api_timings[name])

    per_api_agg = {}
    for name, vals in per_api.items():
        if vals:
            per_api_agg[name] = {
                "count": len(vals),
                "mean_s": round(sum(vals) / len(vals), 4),
                "total_s": round(sum(vals), 4),
            }
    if per_api_agg:
        agg["per_api_latency"] = per_api_agg

    return agg


def aggregate_collection_metrics(
    records: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Per-collection aggregate performance statistics.

    Groups records by ``collection_id`` and computes per-collection
    latency totals and means.  Returns a dict keyed by collection id.
    """
    by_collection: Dict[str, List[float]] = {}
    for r in records:
        cid = r.get("collection_id", "unknown")
        perf = r.get("perf", {})
        wt = perf.get("wall_time_seconds", 0)
        by_collection.setdefault(cid, []).append(wt)

    result: Dict[str, Dict[str, Any]] = {}
    for cid, times in sorted(by_collection.items()):
        n = len(times)
        result[cid] = {
            "items": n,
            "total_s": round(sum(times), 2),
            "mean_s": round(sum(times) / n, 4) if n else 0,
        }
    return result
