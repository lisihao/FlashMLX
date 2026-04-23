"""
FlashMLX Profile Report — Structured JSON output, console display, and run comparison.
"""

import json
import platform
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import mlx.core as mx

from .metrics import MetricCollector


class ProfileReport:
    """Collect metrics from all 8 collectors and produce a structured report."""

    def __init__(self, experiment_name: str, model_name: str = ""):
        self.name = experiment_name
        self.model_name = model_name
        self.metrics: Dict[str, dict] = {}
        self.timestamp = datetime.now().isoformat()
        self._build_metadata()

    def _build_metadata(self):
        metal_mb = 0.0
        try:
            metal_mb = mx.get_active_memory() / (1024 * 1024)
        except Exception:
            pass

        self.metadata = {
            "experiment": self.name,
            "model": self.model_name,
            "timestamp": self.timestamp,
            "platform": platform.platform(),
            "processor": platform.processor(),
            "metal_memory_mb": metal_mb,
        }

    def add_metric(self, name: str, collector: MetricCollector):
        self.metrics[name] = collector.to_dict()

    def add_raw(self, name: str, data: dict):
        self.metrics[name] = data

    def save_json(self, path: str):
        out = {"metadata": self.metadata, "metrics": self.metrics}
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w") as f:
            json.dump(out, f, indent=2, default=str)
        print(f"Report saved: {p}")

    @classmethod
    def load_json(cls, path: str) -> "ProfileReport":
        with open(path) as f:
            data = json.load(f)
        report = cls(
            data["metadata"].get("experiment", "loaded"),
            data["metadata"].get("model", ""),
        )
        report.metadata = data["metadata"]
        report.metrics = data.get("metrics", {})
        return report

    # ---- Console pretty-print ------------------------------------------------

    def print_console(self):
        w = 64
        print()
        print(f"{'=' * w}")
        print(f" FlashMLX Profile: {self.model_name or self.name}")
        print(f" {self.timestamp}")
        print(f"{'=' * w}")

        self._print_kernel_launches(w)
        self._print_moe(w)
        self._print_dequant(w)
        self._print_gpu_gaps(w)
        self._print_batch_scaling(w)
        self._print_context_split(w)

        print(f"{'=' * w}")
        print()

    def _print_kernel_launches(self, w: int):
        m = self.metrics.get("kernel_launches")
        if not m:
            return
        s = m.get("stats", {})
        print(f" 1. Kernel launches/token: "
              f"{s.get('mean', 0):.1f} mean, "
              f"{s.get('min', 0):.0f}-{s.get('max', 0):.0f} range")

    def _print_moe(self, w: int):
        m = self.metrics.get("moe")
        if not m:
            return
        ratio = m.get("moe_time_ratio", {})
        print(f" 2. MoE path time ratio:   {ratio.get('ratio', 0) * 100:.1f}%"
              f"  ({ratio.get('total_moe_ms', 0):.1f}ms / {ratio.get('total_forward_ms', 0):.1f}ms)")

        hist = m.get("expert_histogram", {})
        if hist:
            vals = list(hist.values())
            total = sum(vals)
            top5 = sorted(hist.items(), key=lambda x: x[1], reverse=True)[:5]
            dist_str = ", ".join(f"E{k}:{v}" for k, v in top5)
            print(f" 3. Expert distribution:   {dist_str} (total {total})")

        gemm = m.get("gemm_groups", {})
        gs = gemm.get("stats", {})
        if gs.get("count", 0) > 0:
            print(f" 4. GEMM groups:           avg {gs.get('mean', 0):.1f}, "
                  f"range [{gs.get('min', 0):.0f}, {gs.get('max', 0):.0f}], "
                  f"total {gemm.get('total_groups', 0)}")

    def _print_dequant(self, w: int):
        m = self.metrics.get("dequant")
        if not m:
            return
        print(f" 5. Dequant time ratio:    {m.get('ratio', 0) * 100:.1f}%"
              f"  ({m.get('total_dequant_ms', 0):.1f}ms, "
              f"{m.get('num_dequant_calls', 0)} calls)")

    def _print_gpu_gaps(self, w: int):
        m = self.metrics.get("gpu_gaps")
        if not m or m.get("status") == "not_collected":
            print(f" 6. GPU gaps:              (not collected)")
            return
        print(f" 6. GPU gaps:              {m.get('total_gap_ms', 0):.1f}ms total"
              f"  ({m.get('gap_ratio', 0) * 100:.1f}%)")

    def _print_batch_scaling(self, w: int):
        m = self.metrics.get("batch_scaling")
        if not m:
            return
        results = m.get("results", {})
        if not results:
            return
        parts = []
        for bs in sorted(results.keys(), key=lambda x: int(x)):
            tps = results[bs].get("tok_per_sec", 0)
            parts.append(f"{bs}:{tps:.0f}")
        print(f" 7. Batch scaling (t/s):   {' -> '.join(parts)}")

    def _print_context_split(self, w: int):
        m = self.metrics.get("context_split")
        if not m:
            return
        results = m.get("results", {})
        if not results:
            return
        for ctx in sorted(results.keys(), key=lambda x: int(x)):
            r = results[ctx]
            if r.get("status") == "failed":
                print(f" 8. Context {ctx}: FAILED")
                continue
            print(f" 8. Context {ctx}:  "
                  f"PF {r.get('prefill_ms', 0):.0f}ms, "
                  f"DEC {r.get('decode_mean_ms_per_token', 0):.1f}ms/tok "
                  f"({r.get('decode_tok_per_sec', 0):.0f} t/s), "
                  f"MEM {r.get('metal_memory_mb', 0):.0f}MB")

    # ---- Comparison -----------------------------------------------------------

    @staticmethod
    def compare(before: "ProfileReport", after: "ProfileReport") -> dict:
        """Compare two reports metric by metric."""
        result = {
            "before": before.metadata,
            "after": after.metadata,
            "deltas": {},
        }

        # Kernel launches
        b_kl = before.metrics.get("kernel_launches", {}).get("stats", {})
        a_kl = after.metrics.get("kernel_launches", {}).get("stats", {})
        if b_kl and a_kl:
            result["deltas"]["kernel_launches_mean"] = {
                "before": b_kl.get("mean", 0),
                "after": a_kl.get("mean", 0),
                "delta": a_kl.get("mean", 0) - b_kl.get("mean", 0),
            }

        # MoE ratio
        b_moe = before.metrics.get("moe", {}).get("moe_time_ratio", {})
        a_moe = after.metrics.get("moe", {}).get("moe_time_ratio", {})
        if b_moe and a_moe:
            result["deltas"]["moe_ratio"] = {
                "before": b_moe.get("ratio", 0),
                "after": a_moe.get("ratio", 0),
                "delta": a_moe.get("ratio", 0) - b_moe.get("ratio", 0),
            }

        # Dequant ratio
        b_dq = before.metrics.get("dequant", {})
        a_dq = after.metrics.get("dequant", {})
        if b_dq and a_dq:
            result["deltas"]["dequant_ratio"] = {
                "before": b_dq.get("ratio", 0),
                "after": a_dq.get("ratio", 0),
                "delta": a_dq.get("ratio", 0) - b_dq.get("ratio", 0),
            }

        # Batch scaling
        b_bs = before.metrics.get("batch_scaling", {}).get("results", {})
        a_bs = after.metrics.get("batch_scaling", {}).get("results", {})
        if b_bs and a_bs:
            bs_deltas = {}
            for bs_key in set(list(b_bs.keys()) + list(a_bs.keys())):
                b_tps = b_bs.get(bs_key, {}).get("tok_per_sec", 0)
                a_tps = a_bs.get(bs_key, {}).get("tok_per_sec", 0)
                pct = ((a_tps - b_tps) / b_tps * 100) if b_tps > 0 else 0
                bs_deltas[bs_key] = {
                    "before_tps": b_tps,
                    "after_tps": a_tps,
                    "delta_pct": pct,
                }
            result["deltas"]["batch_scaling"] = bs_deltas

        return result

    @staticmethod
    def print_comparison(comp: dict):
        """Pretty-print a comparison dict."""
        print()
        print("=" * 64)
        print(" Profile Comparison")
        print(f" Before: {comp['before'].get('experiment', '?')}"
              f" ({comp['before'].get('timestamp', '?')})")
        print(f" After:  {comp['after'].get('experiment', '?')}"
              f" ({comp['after'].get('timestamp', '?')})")
        print("=" * 64)

        for key, delta in comp.get("deltas", {}).items():
            if isinstance(delta, dict) and "before" in delta:
                sign = "+" if delta.get("delta", 0) >= 0 else ""
                print(f"  {key}: {delta['before']:.4f} -> {delta['after']:.4f}"
                      f" ({sign}{delta['delta']:.4f})")
            elif isinstance(delta, dict):
                print(f"  {key}:")
                for sub_k, sub_v in delta.items():
                    if isinstance(sub_v, dict):
                        pct = sub_v.get("delta_pct", 0)
                        sign = "+" if pct >= 0 else ""
                        print(f"    batch {sub_k}: "
                              f"{sub_v.get('before_tps', 0):.0f} -> "
                              f"{sub_v.get('after_tps', 0):.0f} t/s "
                              f"({sign}{pct:.1f}%)")

        print("=" * 64)
        print()
