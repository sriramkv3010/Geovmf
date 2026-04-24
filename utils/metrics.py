"""
utils/metrics.py
================
MetricLogger — per-epoch metric tracking with over-fitting and
κ-explosion warnings, formatted summary tables, and best-epoch retrieval.
"""

from __future__ import annotations
from config import OVERFIT_GAP, KAPPA_MAX


class MetricLogger:
    """
    Collects per-epoch scalar metrics, prints live summaries with
    automated warnings, and exposes table / best-row helpers.

    Parameters
    ----------
    stage : str
        Human-readable label (e.g. ``"vMF-frozen"``, ``"joint-CNN-vMF"``).

    Example
    -------
    >>> logger = MetricLogger("vMF-frozen")
    >>> logger.log(1, train_nll=0.82, val_nll=0.91, val_ang_err=14.2,
    ...            kappa_mean=3.1, kappa_min=0.5, kappa_max=12.4)
    >>> logger.summary_table()
    >>> best = logger.best("val_ang_err")
    """

    def __init__(self, stage: str) -> None:
        self.stage   = stage
        self.history: list[dict] = []

    # ── Logging ───────────────────────────────────────────────────────────────

    def log(self, ep: int, **kw) -> None:
        """
        Record metrics for epoch ``ep`` and print a formatted line with
        optional warnings.

        Parameters
        ----------
        ep  : int    — epoch number (1-based)
        **kw        — arbitrary scalar metrics
        """
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

        # Over-fitting guard
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

    # ── Summary table ─────────────────────────────────────────────────────────

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

    # ── Best epoch retrieval ──────────────────────────────────────────────────

    def best(self, key: str = "val_ang_err", mode: str = "min") -> dict:
        """
        Return the history record with the best value for ``key``.

        Parameters
        ----------
        key  : str   — metric name to optimise
        mode : str   — ``"min"`` or ``"max"``

        Returns
        -------
        record : dict  — epoch record, empty if no history
        """
        if not self.history:
            return {}
        fn = min if mode == "min" else max
        return fn(self.history,
                  key=lambda r: r.get(key, float("inf")))
