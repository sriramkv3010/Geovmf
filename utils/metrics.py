from __future__ import annotations
from config import OVERFIT_GAP, KAPPA_MAX


class MetricLogger:

    def __init__(self, stage: str) -> None:
        self.stage   = stage
        self.history: list[dict] = []

    def log(self, ep: int, **kw) -> None:
        self.history.append({"ep": ep, **kw})

        parts = [f"  [{self.stage}] ep{ep}"]
        for k, v in kw.items():
            parts.append(
                f"{k}={'%.4f' % v if isinstance(v, float) else v}")

        warns = self._check_warns(kw)
        line  = " | ".join(parts)
        if warns:
            line += "  " + " ".join(warns)
        print(line)

    def _check_warns(self, kw: dict) -> list[str]:
        warns = []

        if "train_nll" in kw and "val_nll" in kw:
            gap = kw["val_nll"] - kw["train_nll"]
            if gap > OVERFIT_GAP:
                warns.append(f"⚠ OVERFIT gap={gap:.3f}")

        # Angular error regression
        if "val_ang_err" in kw and len(self.history) >= 3:
            prev = self.history[-2].get("val_ang_err", float("inf"))
            if kw["val_ang_err"] > prev + 0.5:
                warns.append("⚠ ANG_ERR ROSE — possible overconfidence")

        # κ ceiling warning
        if "kappa_mean" in kw and kw["kappa_mean"] > KAPPA_MAX * 0.9:
            warns.append(f"⚠ κ near ceiling ({kw['kappa_mean']:.1f})")

        return warns

    def summary_table(self) -> None:
        """Print a formatted epoch-by-epoch summary table to stdout."""
        if not self.history:
            return

        keys   = [k for k in self.history[0] if k != "ep"]
        header = f"{'ep':>4} " + " ".join(f"{k:>13}" for k in keys)

        print(f"\n  ── {self.stage} summary ──")
        print(f"  {header}")
        print(f"  {'─' * len(header)}")

        for r in self.history:
            vals = f"{r['ep']:>4} " + " ".join(
                f"{r.get(k, float('nan')):>13.4f}"
                if isinstance(r.get(k), float)
                else f"{str(r.get(k, '?')):>13}"
                for k in keys)
            print(f"  {vals}")
        print()

    def best(self, key: str = "val_ang_err", mode: str = "min") -> dict:
        if not self.history:
            return {}
        fn = min if mode == "min" else max
        return fn(self.history,
                  key=lambda r: r.get(key, float("inf")))
