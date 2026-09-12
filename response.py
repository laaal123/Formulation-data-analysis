"""Prompt 3 -- response construction.

One response at a time. Profiles are reduced to interpretable scalars rather than
modelled time point by time point, because eight sets of p-values from one curve
is multiple testing wearing a lab coat.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit

from errors import Refusal
from loader import BatchDataset

PROFILE_METRICS = (
    "weibull_td",
    "weibull_beta",
    "first_order_k",
    "mdt",
    "dissolution_efficiency",
    "t50",
    "t80",
    "f2_vs_reference",
    "cross_medium_mdt_spread",
)


# ---------------------------------------------------------------------
# profile metrics
# ---------------------------------------------------------------------
def _weibull(t, td, beta):
    return 100.0 * (1.0 - np.exp(-((t / td) ** beta)))


def fit_weibull(t: np.ndarray, y: np.ndarray) -> Dict[str, float]:
    """Deterministic Weibull fit. Returns NaNs with a reason rather than guessing."""
    m = np.isfinite(t) & np.isfinite(y)
    t, y = np.asarray(t, float)[m], np.asarray(y, float)[m]
    if len(t) < 3 or np.nanmax(y) <= 0:
        return {"weibull_td": np.nan, "weibull_beta": np.nan, "weibull_r2": np.nan,
                "reason": "fewer than 3 usable points"}
    order = np.argsort(t)
    t, y = t[order], y[order]
    try:
        popt, _ = curve_fit(
            _weibull, t, y, p0=[max(np.median(t), 1e-3), 1.0],
            bounds=([1e-4, 0.05], [1e4, 10.0]), maxfev=20000,
        )
    except Exception as exc:  # noqa: BLE001 - report, never silently substitute
        return {"weibull_td": np.nan, "weibull_beta": np.nan, "weibull_r2": np.nan,
                "reason": f"fit failed: {type(exc).__name__}"}
    pred = _weibull(t, *popt)
    ss_res = float(((y - pred) ** 2).sum())
    ss_tot = float(((y - y.mean()) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan
    return {"weibull_td": float(popt[0]), "weibull_beta": float(popt[1]),
            "weibull_r2": float(r2) if np.isfinite(r2) else np.nan, "reason": ""}


def first_order_k(t: np.ndarray, y: np.ndarray) -> float:
    """Slope of -ln(1 - M/100) against t through the origin."""
    m = np.isfinite(t) & np.isfinite(y) & (y < 99.5) & (y > 0)
    if m.sum() < 3:
        return np.nan
    tt, yy = np.asarray(t, float)[m], np.asarray(y, float)[m]
    z = -np.log(1.0 - yy / 100.0)
    denom = float((tt**2).sum())
    return float((tt * z).sum() / denom) if denom > 0 else np.nan


def mean_dissolution_time(t: np.ndarray, y: np.ndarray) -> float:
    """Model-independent MDT from the increments, with midpoint times."""
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    tt, yy = np.asarray(t, float)[m], np.asarray(y, float)[m]
    order = np.argsort(tt)
    tt, yy = tt[order], yy[order]
    t_prev = np.concatenate([[0.0], tt[:-1]])
    y_prev = np.concatenate([[0.0], yy[:-1]])
    dy = yy - y_prev
    t_mid = (tt + t_prev) / 2.0
    total = float(dy.sum())
    if total <= 0:
        return np.nan
    return float((t_mid * dy).sum() / total)


def dissolution_efficiency(t: np.ndarray, y: np.ndarray) -> float:
    """Area under the curve as a percentage of the rectangle of total release."""
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    tt, yy = np.asarray(t, float)[m], np.asarray(y, float)[m]
    order = np.argsort(tt)
    tt, yy = tt[order], yy[order]
    tt = np.concatenate([[0.0], tt])
    yy = np.concatenate([[0.0], yy])
    auc = float(np.trapezoid(yy, tt)) if hasattr(np, "trapezoid") else float(np.trapz(yy, tt))
    span = float(tt[-1] - tt[0])
    return float(auc / (span * 100.0) * 100.0) if span > 0 else np.nan


def time_to_percent(t: np.ndarray, y: np.ndarray, target: float) -> float:
    """Interpolated time to reach `target` % released. NaN if never reached -- the
    value is censored, and extrapolating it would invent data."""
    m = np.isfinite(t) & np.isfinite(y)
    if m.sum() < 2:
        return np.nan
    tt, yy = np.asarray(t, float)[m], np.asarray(y, float)[m]
    order = np.argsort(tt)
    tt, yy = tt[order], yy[order]
    if yy.max() < target:
        return np.nan
    idx = int(np.argmax(yy >= target))
    if idx == 0:
        return float(tt[0] * target / yy[0]) if yy[0] > 0 else float(tt[0])
    y0, y1 = yy[idx - 1], yy[idx]
    t0, t1 = tt[idx - 1], tt[idx]
    if y1 == y0:
        return float(t1)
    return float(t0 + (target - y0) * (t1 - t0) / (y1 - y0))


def f2_similarity(reference: np.ndarray, test: np.ndarray) -> float:
    """f2 on paired time points. Returns NaN if fewer than 3 paired points."""
    m = np.isfinite(reference) & np.isfinite(test)
    if m.sum() < 3:
        return np.nan
    r, s = np.asarray(reference, float)[m], np.asarray(test, float)[m]
    diff = float(((r - s) ** 2).mean())
    return float(50.0 * np.log10(100.0 / np.sqrt(1.0 + diff)))


def profile_metrics_table(diss: pd.DataFrame) -> pd.DataFrame:
    """One row per (batch, medium) with every profile scalar."""
    rows = []
    for (batch, medium), g in diss.groupby(["batch", "medium"], sort=True):
        g = g.sort_values("time_h")
        t = g["time_h"].to_numpy(dtype=float)
        y = g["percent_released"].to_numpy(dtype=float)
        w = fit_weibull(t, y)
        rows.append(
            {
                "batch": batch,
                "medium": medium,
                "weibull_td": w["weibull_td"],
                "weibull_beta": w["weibull_beta"],
                "weibull_r2": w["weibull_r2"],
                "weibull_note": w["reason"],
                "first_order_k": first_order_k(t, y),
                "mdt": mean_dissolution_time(t, y),
                "dissolution_efficiency": dissolution_efficiency(t, y),
                "t50": time_to_percent(t, y, 50.0),
                "t80": time_to_percent(t, y, 80.0),
                "n_time_points": int(len(t)),
            }
        )
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------
# response object
# ---------------------------------------------------------------------
@dataclass
class ResponseResult:
    name: str
    values: pd.Series                     # indexed by batch
    kind: str                             # "scalar" or "profile_metric"
    source: Dict = field(default_factory=dict)
    diagnostics: Dict = field(default_factory=dict)
    statements: List[str] = field(default_factory=list)
    usable: bool = True

    def text(self) -> str:
        return "\n".join(self.statements)


def _signal_vs_noise(values: pd.Series, formulations: pd.Series) -> Dict:
    """One-way variance components: between-formulation against within-formulation."""
    df = pd.DataFrame({"y": pd.to_numeric(values, errors="coerce"), "g": formulations}).dropna()
    out = {
        "n": int(len(df)),
        "n_groups": int(df["g"].nunique()),
        "sd_total": float(df["y"].std(ddof=1)) if len(df) > 1 else np.nan,
    }
    if len(df) < 3 or df["g"].nunique() < 2:
        out.update(ms_between=np.nan, ms_within=np.nan, f_ratio=np.nan, icc=np.nan,
                   sd_within=np.nan, sd_between=np.nan)
        return out
    grand = df["y"].mean()
    groups = df.groupby("g")["y"]
    k = df["g"].nunique()
    n_total = len(df)
    ss_between = float(sum(len(s) * (s.mean() - grand) ** 2 for _, s in groups))
    ss_within = float(sum(((s - s.mean()) ** 2).sum() for _, s in groups))
    df_b, df_w = k - 1, n_total - k
    ms_b = ss_between / df_b if df_b > 0 else np.nan
    ms_w = ss_within / df_w if df_w > 0 else np.nan
    sizes = groups.size().to_numpy(dtype=float)
    n0 = (n_total - (sizes**2).sum() / n_total) / (k - 1) if k > 1 else np.nan
    var_b = (ms_b - ms_w) / n0 if np.isfinite(ms_b) and np.isfinite(ms_w) and n0 else np.nan
    var_b = max(var_b, 0.0) if np.isfinite(var_b) else np.nan
    icc = var_b / (var_b + ms_w) if np.isfinite(var_b) and np.isfinite(ms_w) and (var_b + ms_w) > 0 else np.nan
    out.update(
        ms_between=ms_b,
        ms_within=ms_w,
        f_ratio=(ms_b / ms_w) if (np.isfinite(ms_w) and ms_w > 0) else np.nan,
        icc=icc,
        sd_within=float(np.sqrt(ms_w)) if np.isfinite(ms_w) else np.nan,
        sd_between=float(np.sqrt(var_b)) if np.isfinite(var_b) else np.nan,
    )
    return out


def build_response(
    ds: BatchDataset,
    name: Optional[str] = None,
    *,
    metric: Optional[str] = None,
    medium: Optional[str] = None,
    reference_batch: Optional[str] = None,
    method_rsd_pct: Optional[float] = None,
    units: str = "",
) -> ResponseResult:
    """Build one response.

    Either `name` = a RESPONSE column in the wide table, or `metric` = one of
    PROFILE_METRICS computed from the dissolution table.
    """
    statements: List[str] = []
    if metric is not None:
        values, source = _profile_response(ds, metric, medium, reference_batch)
        kind = "profile_metric"
        label = f"{metric}" + (f" [{medium}]" if medium else "")
    else:
        if name is None:
            raise ValueError("Provide either a response column name or a profile metric")
        if name not in ds.wide.columns:
            raise KeyError(f"{name!r} is not a column in the batch table")
        if ds.families.get(name) != "RESPONSE":
            statements.append(
                f"Note: {name!r} is tagged {ds.families.get(name)}, not RESPONSE. It is being "
                "used as the response anyway; make sure it is excluded from the predictors."
            )
        values = pd.to_numeric(ds.wide[name], errors="coerce")
        source = {"column": name}
        kind = "scalar"
        label = name

    values = values.reindex(ds.wide.index)
    n_missing = int(values.isna().sum())
    usable = True

    if n_missing:
        statements.append(
            f"{n_missing} of {len(values)} batches have no value for this response "
            "(censored or not measured). Those batches are excluded from modelling, and "
            "that exclusion is not random - check whether it is related to the response "
            "itself before trusting anything downstream."
        )
    observed = values.dropna()
    if len(observed) < 3:
        statements.append("Fewer than 3 batches have this response. Nothing can be modelled.")
        usable = False

    diagnostics: Dict = {
        "n_observed": int(len(observed)),
        "n_missing": n_missing,
        "min": float(observed.min()) if len(observed) else np.nan,
        "max": float(observed.max()) if len(observed) else np.nan,
        "mean": float(observed.mean()) if len(observed) else np.nan,
        "sd": float(observed.std(ddof=1)) if len(observed) > 1 else np.nan,
        "range": float(observed.max() - observed.min()) if len(observed) else np.nan,
    }

    if len(observed) > 1 and observed.nunique() == 1:
        statements.append(
            "This response takes a single value in every batch. There is no variation to "
            "attribute. Refusing to model it."
        )
        usable = False

    var = _signal_vs_noise(values, ds.formulations)
    diagnostics["variance_components"] = var
    if np.isfinite(var.get("f_ratio", np.nan)):
        if var["f_ratio"] < 1.0:
            statements.append(
                f"Between-batch noise swamps the between-formulation signal "
                f"(within-formulation SD {var['sd_within']:.3g} against between-formulation "
                f"SD {var['sd_between'] if np.isfinite(var['sd_between']) else 0:.3g}, F = "
                f"{var['f_ratio']:.2f}). There is nothing here to attribute to formulation "
                "differences; any factor that only changes between formulations cannot be "
                "assessed against this response."
            )
        else:
            statements.append(
                f"Between-formulation signal exceeds within-formulation noise "
                f"(F = {var['f_ratio']:.2f}, ICC = {var['icc']:.2f}); "
                f"within-formulation SD {var['sd_within']:.3g}, between-formulation SD "
                f"{var['sd_between']:.3g}."
            )

    if method_rsd_pct is not None and len(observed) > 1:
        mean = diagnostics["mean"]
        method_sd = abs(method_rsd_pct) / 100.0 * abs(mean) if np.isfinite(mean) else np.nan
        diagnostics["method_sd"] = float(method_sd)
        diagnostics["method_rsd_pct"] = float(method_rsd_pct)
        ratio = diagnostics["range"] / method_sd if method_sd and np.isfinite(method_sd) else np.nan
        diagnostics["range_over_method_sd"] = float(ratio) if np.isfinite(ratio) else np.nan
        if np.isfinite(ratio) and ratio < 3:
            statements.append(
                f"The whole observed range of this response ({diagnostics['range']:.3g}) is "
                f"less than three times the analytical method's own SD ({method_sd:.3g} from "
                f"an RSD of {method_rsd_pct:.2f}%). Attributing differences of this size is "
                "chasing measurement noise. Fix the method or widen the range before modelling."
            )
            usable = False
        elif np.isfinite(ratio):
            statements.append(
                f"Observed range is {ratio:.1f} times the analytical method SD "
                f"({method_sd:.3g}), so the variation is larger than method noise."
            )

    return ResponseResult(
        name=label,
        values=values,
        kind=kind,
        source=source,
        diagnostics=diagnostics,
        statements=statements,
        usable=usable,
    )


def _profile_response(
    ds: BatchDataset, metric: str, medium: Optional[str], reference_batch: Optional[str]
):
    if ds.dissolution is None:
        raise ValueError("No dissolution table was loaded")
    if metric not in PROFILE_METRICS:
        raise ValueError(f"metric must be one of {PROFILE_METRICS}")
    diss = ds.dissolution

    if metric == "cross_medium_mdt_spread":
        table = profile_metrics_table(diss)
        wide = table.pivot(index="batch", columns="medium", values="mdt")
        if wide.shape[1] < 2:
            raise ValueError("Cross-medium spread needs at least two media")
        values = wide.max(axis=1) - wide.min(axis=1)
        return values, {"metric": metric, "media": list(wide.columns)}

    if medium is not None:
        diss = diss[diss["medium"] == medium]
        if diss.empty:
            raise ValueError(f"No dissolution rows for medium {medium!r}")
    elif diss["medium"].nunique() > 1:
        raise ValueError(
            "Several media present; choose one with medium=, or use "
            "'cross_medium_mdt_spread'."
        )

    if metric == "f2_vs_reference":
        if reference_batch is None:
            raise ValueError("f2 needs a reference batch")
        ref = (
            diss[diss["batch"] == reference_batch]
            .sort_values("time_h")
            .set_index("time_h")["percent_released"]
        )
        if ref.empty:
            raise ValueError(f"Reference batch {reference_batch!r} has no dissolution data")
        vals = {}
        for batch, g in diss.groupby("batch"):
            s = g.sort_values("time_h").set_index("time_h")["percent_released"]
            common = ref.index.intersection(s.index)
            vals[batch] = f2_similarity(ref.loc[common].to_numpy(), s.loc[common].to_numpy())
        values = pd.Series(vals, dtype=float)
        return values, {"metric": metric, "medium": medium, "reference": reference_batch}

    table = profile_metrics_table(diss)
    if metric not in table.columns:
        raise ValueError(f"{metric} not available")
    values = table.set_index("batch")[metric].astype(float)
    return values, {"metric": metric, "medium": medium}
