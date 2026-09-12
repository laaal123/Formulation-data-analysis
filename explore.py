"""Analyses that are honest at small n.

Attribution needs many batches. Comparison, profile similarity and "what differs
between these two batches" do not: they describe what is in front of you rather
than inferring what caused it, so they are valid from two batches upward.

Nothing in this module fits a model or ranks causes.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from errors import Refusal
from loader import BatchDataset
from response import profile_metrics_table


# ---------------------------------------------------------------------
# side-by-side comparison
# ---------------------------------------------------------------------
def compare_batches(
    ds: BatchDataset, batches: Optional[Iterable[str]] = None, families: Optional[Iterable[str]] = None
) -> pd.DataFrame:
    """Every variable, one column per batch. The plainest possible view."""
    sel = list(batches) if batches else list(ds.wide.index)
    missing = [b for b in sel if b not in ds.wide.index]
    if missing:
        raise Refusal(f"Not in the data: {', '.join(map(str, missing))}")
    if len(sel) < 1:
        raise Refusal("Select at least one batch to compare.")
    cols = [
        c for c in ds.wide.columns
        if families is None or ds.families.get(c) in set(families)
    ]
    out = ds.wide.loc[sel, cols].T
    out.index.name = "variable"
    out.insert(0, "family", [ds.families.get(c, "UNTAGGED") for c in out.index])
    return out


# ---------------------------------------------------------------------
# what differs between two batches
# ---------------------------------------------------------------------
def what_changed(
    ds: BatchDataset,
    batch_a: str,
    batch_b: str,
    min_pct: float = 0.0,
) -> pd.DataFrame:
    """Every numeric variable that differs between two batches, largest first.

    This is a difference list, not an attribution. It tells you what is not the
    same; it cannot tell you which of those differences matters. With only two
    batches that distinction is the whole story, because every difference is
    perfectly confounded with every other.
    """
    for b in (batch_a, batch_b):
        if b not in ds.wide.index:
            raise Refusal(f"Batch {b!r} is not in the data.")
    if batch_a == batch_b:
        raise Refusal("Choose two different batches.")

    a = ds.wide.loc[batch_a]
    b = ds.wide.loc[batch_b]
    # spread across all batches, used only to say whether a gap is unusual
    have_spread = ds.n_batches >= 5
    rows = []
    for col in ds.wide.columns:
        va, vb = pd.to_numeric(a.get(col), errors="coerce"), pd.to_numeric(b.get(col), errors="coerce")
        if not (np.isfinite(va) and np.isfinite(vb)):
            continue
        diff = float(vb - va)
        base = abs(float(va)) if abs(float(va)) > 1e-12 else np.nan
        pct = 100.0 * diff / base if np.isfinite(base) else np.nan
        sd = float(pd.to_numeric(ds.wide[col], errors="coerce").std(ddof=1)) if have_spread else np.nan
        z = diff / sd if (have_spread and np.isfinite(sd) and sd > 0) else np.nan
        rows.append(
            {
                "variable": col,
                "family": ds.families.get(col, "UNTAGGED"),
                batch_a: float(va),
                batch_b: float(vb),
                "difference": diff,
                "percent_change": pct,
                "sd_across_all_batches": sd,
                "difference_in_sd": z,
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    if min_pct:
        out = out[out["percent_change"].abs() >= min_pct]
    key = "difference_in_sd" if have_spread else "percent_change"
    out = out.reindex(out[key].abs().sort_values(ascending=False).index)
    caption = (
        f"{len(out)} variables differ between {batch_a} and {batch_b}. Every one of them "
        "changed at the same time, so none can be credited with the difference in any "
        "response. Use this to decide what to test, not what to conclude."
    )
    if have_spread:
        p = int(len(out))
        # two-sided normal tail, purely as an expectation of how many large gaps
        # would appear even if nothing systematic had changed
        exp2 = p * 0.0455
        exp3 = p * 0.0027
        big2 = int((out["difference_in_sd"].abs() > 2).sum())
        big3 = int((out["difference_in_sd"].abs() > 3).sum())
        caption += (
            f" Across {p} variables you would expect roughly {exp2:.1f} to differ by more "
            f"than 2 SD and {exp3:.1f} by more than 3 SD by chance alone; you have "
            f"{big2} and {big3}. Treat the top of this list as candidates, not as evidence, "
            "and pay more attention to variables you deliberately changed than to ones you "
            "merely recorded."
        )
    out.attrs["caption"] = caption
    return out.reset_index(drop=True)


# ---------------------------------------------------------------------
# dissolution profile work
# ---------------------------------------------------------------------
def f1_difference(reference: np.ndarray, test: np.ndarray) -> float:
    """f1 difference factor. Below 15 is the conventional similarity limit."""
    m = np.isfinite(reference) & np.isfinite(test)
    if m.sum() < 3:
        return np.nan
    r, t = np.asarray(reference, float)[m], np.asarray(test, float)[m]
    denom = float(np.sum(r))
    if denom <= 0:
        return np.nan
    return float(100.0 * np.sum(np.abs(r - t)) / denom)


def compare_profiles(
    ds: BatchDataset, reference_batch: str, test_batch: str, medium: Optional[str] = None
) -> Dict:
    """f2, f1 and the point-by-point differences for two batches.

    Valid with exactly two batches. This is the standard regulatory comparison and it
    does not need a model.
    """
    from response import f2_similarity

    if ds.dissolution is None:
        raise Refusal("No dissolution data was loaded.")
    diss = ds.dissolution
    if medium is not None:
        diss = diss[diss["medium"] == medium]
    elif diss["medium"].nunique() > 1:
        raise Refusal("Several media are present; choose one.")
    if diss.empty:
        raise Refusal("No dissolution rows for that medium.")

    ref = diss[diss["batch"] == reference_batch].sort_values("time_h").set_index("time_h")["percent_released"]
    tst = diss[diss["batch"] == test_batch].sort_values("time_h").set_index("time_h")["percent_released"]
    if ref.empty or tst.empty:
        raise Refusal("One of those batches has no dissolution data in this medium.")
    common = ref.index.intersection(tst.index)
    if len(common) < 3:
        raise Refusal(
            f"Only {len(common)} shared time point(s). f2 needs at least three, and the "
            "guidance expects at most one point above 85% released."
        )
    r, t = ref.loc[common].to_numpy(float), tst.loc[common].to_numpy(float)
    f2 = f2_similarity(r, t)
    f1 = f1_difference(r, t)
    table = pd.DataFrame(
        {"time_h": common, reference_batch: r, test_batch: t, "difference": t - r}
    )
    above85 = int(np.sum((r > 85) & (t > 85)))
    notes = []
    if above85 > 1:
        notes.append(
            f"{above85} time points have both profiles above 85% released. Guidance allows "
            "only one; f2 is inflated by the rest and should not be quoted as it stands."
        )
    verdict = (
        "similar by the conventional f2 >= 50 criterion"
        if np.isfinite(f2) and f2 >= 50
        else "not similar by the conventional f2 >= 50 criterion"
    )
    return {
        "f2": f2,
        "f1": f1,
        "n_points": int(len(common)),
        "table": table,
        "verdict": verdict,
        "notes": notes,
        "medium": medium,
    }


def profile_summary(ds: BatchDataset, medium: Optional[str] = None) -> pd.DataFrame:
    """Weibull, MDT, dissolution efficiency, t50 and t80 for every batch."""
    if ds.dissolution is None:
        raise Refusal("No dissolution data was loaded.")
    diss = ds.dissolution
    if medium is not None:
        diss = diss[diss["medium"] == medium]
    if diss.empty:
        raise Refusal("No dissolution rows for that medium.")
    return profile_metrics_table(diss)


# ---------------------------------------------------------------------
# trending and specification
# ---------------------------------------------------------------------
def trend_table(ds: BatchDataset, column: str) -> pd.DataFrame:
    """One variable across batches in manufacturing order, with the mean and spread.

    Not a control chart: control limits need a stable process and far more batches
    than a formulation series has. The limits shown are descriptive only and are
    labelled as such.
    """
    if column not in ds.wide.columns:
        raise Refusal(f"{column!r} is not in the data.")
    v = pd.to_numeric(ds.wide[column], errors="coerce")
    order = ds.wide.index
    if "manufacturing_date" in ds.metadata.columns:
        dates = pd.to_datetime(ds.metadata.loc[order, "manufacturing_date"], errors="coerce")
        order = dates.sort_values().index
    out = pd.DataFrame(
        {
            "batch": order,
            "formulation": ds.metadata.loc[order, "formulation"].to_numpy(),
            column: v.reindex(order).to_numpy(),
        }
    )
    if "manufacturing_date" in ds.metadata.columns:
        out["manufacturing_date"] = pd.to_datetime(
            ds.metadata.loc[order, "manufacturing_date"], errors="coerce"
        ).to_numpy()
    obs = out[column].dropna()
    out.attrs["mean"] = float(obs.mean()) if len(obs) else np.nan
    out.attrs["sd"] = float(obs.std(ddof=1)) if len(obs) > 1 else np.nan
    out.attrs["n"] = int(len(obs))
    out.attrs["caption"] = (
        "Descriptive mean and spread. With a handful of batches from a changing "
        "formulation these are not control limits and must not be used as them."
    )
    return out.reset_index(drop=True)


def spec_check(ds: BatchDataset, limits: Dict[str, Tuple[Optional[float], Optional[float]]]) -> pd.DataFrame:
    """Pass or fail each batch against limits you supply, as (lower, upper)."""
    rows = []
    for col, (lo, hi) in limits.items():
        if col not in ds.wide.columns:
            raise Refusal(f"{col!r} is not in the data.")
        v = pd.to_numeric(ds.wide[col], errors="coerce")
        for batch, value in v.items():
            if not np.isfinite(value):
                status = "no result"
            elif lo is not None and value < lo:
                status = "BELOW limit"
            elif hi is not None and value > hi:
                status = "ABOVE limit"
            else:
                status = "within limits"
            rows.append(
                {
                    "batch": batch,
                    "variable": col,
                    "value": float(value) if np.isfinite(value) else np.nan,
                    "lower": lo,
                    "upper": hi,
                    "status": status,
                }
            )
    out = pd.DataFrame(rows)
    out.attrs["caption"] = (
        "A comparison against limits you entered. It is not a release decision and "
        "carries no statistical statement about future batches."
    )
    return out


def descriptive_summary(ds: BatchDataset, columns: Optional[Iterable[str]] = None) -> pd.DataFrame:
    """n, mean, SD, RSD, min, max per variable, by formulation where there is more than one."""
    cols = list(columns) if columns else [
        c for c in ds.wide.columns if pd.api.types.is_numeric_dtype(ds.wide[c])
    ]
    rows = []
    for c in cols:
        v = pd.to_numeric(ds.wide[c], errors="coerce").dropna()
        if v.empty:
            continue
        mean = float(v.mean())
        sd = float(v.std(ddof=1)) if len(v) > 1 else np.nan
        rows.append(
            {
                "variable": c,
                "family": ds.families.get(c, "UNTAGGED"),
                "n": int(len(v)),
                "mean": mean,
                "sd": sd,
                "rsd_pct": (100.0 * sd / abs(mean)) if (np.isfinite(sd) and abs(mean) > 1e-12) else np.nan,
                "min": float(v.min()),
                "max": float(v.max()),
                "range": float(v.max() - v.min()),
            }
        )
    return pd.DataFrame(rows)
