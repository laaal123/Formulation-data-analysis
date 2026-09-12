"""Prompt 2 -- design diagnostics.

Runs before any model. Its output gates modelling: what this dataset can and
cannot answer, in plain sentences.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from config import THRESHOLDS
from errors import Refusal
from loader import BatchDataset


# ---------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------
def _numeric_matrix(ds: BatchDataset, columns: List[str]) -> pd.DataFrame:
    X = ds.wide[columns].apply(pd.to_numeric, errors="coerce")
    return X


def _safe_corr(X: pd.DataFrame) -> pd.DataFrame:
    """Pairwise correlation, NaN where a pair has <3 shared observations or no variance."""
    cols = list(X.columns)
    out = pd.DataFrame(np.nan, index=cols, columns=cols, dtype=float)
    arr = X.to_numpy(dtype=float)
    for i in range(len(cols)):
        out.iat[i, i] = 1.0
        for j in range(i + 1, len(cols)):
            a, b = arr[:, i], arr[:, j]
            m = np.isfinite(a) & np.isfinite(b)
            if m.sum() < 3:
                continue
            av, bv = a[m], b[m]
            if np.std(av) == 0 or np.std(bv) == 0:
                continue
            r = float(np.corrcoef(av, bv)[0, 1])
            out.iat[i, j] = out.iat[j, i] = r
    return out


def _connected_components(adj: Dict[str, set]) -> List[List[str]]:
    seen, comps = set(), []
    for node in adj:
        if node in seen:
            continue
        stack, comp = [node], []
        seen.add(node)
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb in adj[cur]:
                if nb not in seen:
                    seen.add(nb)
                    stack.append(nb)
        comps.append(sorted(comp))
    return comps


def eta_squared_by_group(x: pd.Series, groups: pd.Series) -> float:
    """R-squared of x regressed on group identity (one-way ANOVA R^2)."""
    x = pd.to_numeric(x, errors="coerce")
    df = pd.DataFrame({"x": x, "g": groups.values}).dropna()
    if df.empty or df["x"].std(ddof=0) == 0:
        return float("nan")
    grand = df["x"].mean()
    ss_total = float(((df["x"] - grand) ** 2).sum())
    if ss_total == 0:
        return float("nan")
    ss_between = float(
        df.groupby("g")["x"].apply(lambda s: len(s) * (s.mean() - grand) ** 2).sum()
    )
    return float(np.clip(ss_between / ss_total, 0.0, 1.0))


def r2_on_time(x: pd.Series, dates: pd.Series) -> float:
    x = pd.to_numeric(x, errors="coerce")
    t = pd.to_datetime(dates, errors="coerce")
    df = pd.DataFrame({"x": x.values, "t": t.values}).dropna()
    if len(df) < 3:
        return float("nan")
    tt = df["t"].astype("int64").to_numpy(dtype=float)
    xx = df["x"].to_numpy(dtype=float)
    if np.std(tt) == 0 or np.std(xx) == 0:
        return float("nan")
    r = np.corrcoef(tt, xx)[0, 1]
    return float(r**2)


def variance_inflation_factors(X: pd.DataFrame) -> pd.Series:
    """VIF via R^2 of each column on the rest. Undefined when p >= n-1."""
    Xc = X.dropna(axis=0, how="any")
    n, p = Xc.shape
    if p < 2 or n <= p + 1:
        return pd.Series(np.nan, index=X.columns, dtype=float)
    A = Xc.to_numpy(dtype=float)
    A = A - A.mean(axis=0)
    sd = A.std(axis=0, ddof=0)
    keep = sd > 0
    A = A[:, keep] / sd[keep]
    cols = np.array(Xc.columns)[keep]
    vifs = pd.Series(np.nan, index=X.columns, dtype=float)
    for k in range(A.shape[1]):
        y = A[:, k]
        Z = np.delete(A, k, axis=1)
        Z = np.column_stack([np.ones(len(Z)), Z])
        beta, *_ = np.linalg.lstsq(Z, y, rcond=None)
        resid = y - Z @ beta
        ss_tot = float((y**2).sum())
        r2 = 1.0 - float((resid**2).sum()) / ss_tot if ss_tot > 0 else np.nan
        r2 = min(r2, 1 - 1e-12) if np.isfinite(r2) else np.nan
        vifs[cols[k]] = 1.0 / (1.0 - r2) if np.isfinite(r2) else np.nan
    return vifs


# ---------------------------------------------------------------------
# result object
# ---------------------------------------------------------------------
@dataclass
class DesignAssessment:
    n_batches: int
    n_formulations: int
    n_predictors: int
    p_over_n_flag: bool
    correlation: pd.DataFrame
    clusters: List[List[str]]                      # every predictor, clustered
    cluster_of: Dict[str, str]                     # column -> cluster name
    inseparable_groups: List[List[str]]            # clusters with >1 member
    vif: pd.Series
    formulation_alias_r2: pd.Series
    time_alias_r2: pd.Series
    aliased_with_formulation: List[str]
    aliased_with_time: List[str]
    mixture_groups: List[Dict]
    linear_dependencies: List[Dict]
    range_table: pd.DataFrame
    zero_information: List[str]
    within_formulation_estimable: List[str]
    statements: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    @property
    def cluster_representatives(self) -> List[str]:
        """One representative per cluster: the alphabetically first member, so the
        choice is deterministic and never mistaken for a ranking."""
        return [c[0] for c in self.clusters]

    def cluster_members(self, column: str) -> List[str]:
        name = self.cluster_of.get(column)
        for c in self.clusters:
            if self.cluster_of.get(c[0]) == name:
                return c
        return [column]

    def confounds_for(self, column: str) -> List[str]:
        """Everything this column cannot be separated from."""
        members = [m for m in self.cluster_members(column) if m != column]
        extra = []
        if column in self.aliased_with_formulation:
            extra.append("formulation identity (and everything else that changed with it)")
        if column in self.aliased_with_time:
            extra.append("manufacturing date")
        return members + extra

    def text(self) -> str:
        return "\n".join(self.statements)


# ---------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------
def assess_design(
    ds: BatchDataset,
    predictors: Optional[List[str]] = None,
    cluster_r: float = THRESHOLDS["collinear_cluster_r"],
) -> DesignAssessment:
    cols = list(predictors) if predictors is not None else ds.predictor_columns()
    n = ds.n_batches
    if n < 3:
        raise Refusal(
            f"{n} batch(es) in this dataset. A design cannot be assessed and nothing can be "
            "attributed from fewer than three batches."
        )
    if len(cols) < 1:
        raise Refusal(
            "No candidate predictor carries any information: every column is constant, "
            "entirely missing, or excluded as non-numeric. There is nothing to attribute to. "
            f"Columns excluded at load: {', '.join(sorted(ds.excluded)) or 'none'}."
        )
    X = _numeric_matrix(ds, cols)
    statements: List[str] = []
    notes: List[str] = []

    # exclusions made at load time belong in this assessment too, otherwise a user
    # reading only the diagnostics never learns those columns existed
    if ds.excluded:
        statements.append(
            f"{len(ds.excluded)} column(s) were excluded when the data was loaded and are not "
            "candidates here: "
            + "; ".join(f"{c} ({r})" for c, r in sorted(ds.excluded.items())[:8])
            + ("; ..." if len(ds.excluded) > 8 else "")
            + "."
        )

    # -- 4. replication and range (first: it removes dead columns) -----
    rows = []
    zero_info = []
    for c in cols:
        s = X[c].dropna()
        nunique = int(s.nunique())
        rng_ = float(s.max() - s.min()) if len(s) else float("nan")
        sd = float(s.std(ddof=1)) if len(s) > 1 else float("nan")
        mean = float(s.mean()) if len(s) else float("nan")
        cv = abs(sd / mean) if (np.isfinite(sd) and mean not in (0.0,) and np.isfinite(mean)) else np.nan
        informative = nunique > 1
        rows.append(
            {
                "column": c,
                "n_distinct": nunique,
                "min": float(s.min()) if len(s) else np.nan,
                "max": float(s.max()) if len(s) else np.nan,
                "range": rng_,
                "sd": sd,
                "cv": cv,
                "n_missing": int(X[c].isna().sum()),
                "informative": informative,
            }
        )
        if not informative:
            zero_info.append(c)
    range_table = pd.DataFrame(rows).set_index("column")
    live = [c for c in cols if c not in zero_info]
    if zero_info:
        statements.append(
            f"{len(zero_info)} predictor(s) take a single value in every batch and are "
            f"excluded because they carry no information: {', '.join(zero_info[:8])}"
            + (" ..." if len(zero_info) > 8 else "")
            + "."
        )

    Xl = X[live]
    p = len(live)

    # -- 1. design matrix audit ---------------------------------------
    p_flag = p > n * THRESHOLDS["p_over_n_flag"]
    statements.append(
        f"{n} batches against {p} candidate predictors across {ds.n_formulations} formulations."
    )
    if p_flag:
        statements.append(
            f"p is greater than n/3 ({p} > {n/3:.1f}). Variable selection will be unstable: "
            "different resamples of the same data will select different predictors, and any "
            "single ranking should be read as one draw from a wide distribution."
        )
    if p >= n:
        statements.append(
            f"With p >= n ({p} >= {n}) an unpenalised model fits the data exactly and tells "
            "you nothing. Only penalised or latent-variable models are offered, and only "
            "against a permutation null."
        )

    formulations = ds.formulations
    counts = formulations.value_counts().sort_index()
    statements.append(
        "Batches within a formulation are not independent replicates of the formulation "
        "factor. Effective sample size for anything that changed only between formulations is "
        f"{ds.n_formulations}, not {n} (batches per formulation: "
        + ", ".join(f"{k}={v}" for k, v in counts.items())
        + ")."
    )

    # -- 2. confounding map -------------------------------------------
    corr = _safe_corr(Xl)
    adj: Dict[str, set] = {c: set() for c in live}
    for i, a in enumerate(live):
        for b in live[i + 1 :]:
            r = corr.at[a, b]
            if np.isfinite(r) and abs(r) >= cluster_r:
                adj[a].add(b)
                adj[b].add(a)
    clusters = _connected_components(adj)
    clusters = [sorted(c) for c in clusters]
    clusters.sort(key=lambda c: (-len(c), c[0]))
    cluster_of: Dict[str, str] = {}
    for c in clusters:
        name = c[0] if len(c) == 1 else f"cluster[{c[0]}+{len(c)-1}]"
        for m in c:
            cluster_of[m] = name
    inseparable = [c for c in clusters if len(c) > 1]

    for grp in inseparable:
        pairs = [
            (a, b, corr.at[a, b])
            for i, a in enumerate(grp)
            for b in grp[i + 1 :]
            if np.isfinite(corr.at[a, b]) and abs(corr.at[a, b]) >= cluster_r
        ]
        example = max(pairs, key=lambda t: abs(t[2])) if pairs else None
        msg = (
            f"Cannot be separated: {', '.join(grp)}. These move together across all "
            f"{n} batches"
        )
        if example:
            msg += f" (for example r = {example[2]:.2f} between {example[0]} and {example[1]})"
        msg += (
            ". Any model attributing the response change to one of them rather than another "
            "is making an arbitrary choice, not a finding."
        )
        statements.append(msg)

    # VIF on cluster representatives (full-set VIF is undefined at p >= n-1)
    reps = [c[0] for c in clusters]
    vif_full = variance_inflation_factors(Xl)
    if vif_full.isna().all():
        vif = variance_inflation_factors(Xl[reps])
        if vif.isna().all():
            notes.append(
                f"Variance inflation factors cannot be computed at all: p={p} on the full set "
                f"and {len(reps)} after clustering, both against n-1={n-1}. With more "
                "predictors than batches every predictor is an exact linear combination of "
                "the others, which is itself the finding."
            )
        else:
            notes.append(
                f"Variance inflation factors are undefined on the full predictor set "
                f"(p={p} >= n-1={n-1}); they are reported for the {len(reps)} cluster "
                "representatives, where they are computable."
            )
    else:
        vif = vif_full
    high_vif = sorted(vif[vif > THRESHOLDS["vif_flag"]].index) if vif.notna().any() else []
    if high_vif:
        statements.append(
            f"{len(high_vif)} predictor(s) have VIF above {THRESHOLDS['vif_flag']:.0f} even "
            "after clustering, so their individual coefficients are unstable: "
            + ", ".join(high_vif[:8])
            + ("..." if len(high_vif) > 8 else "")
            + "."
        )

    # aliasing with formulation
    alias_f = pd.Series(
        {c: eta_squared_by_group(Xl[c], formulations) for c in live}, dtype=float
    )
    aliased_f = sorted(alias_f[alias_f > THRESHOLDS["formulation_alias_r2"]].index)
    if aliased_f:
        statements.append(
            f"{len(aliased_f)} predictor(s) are aliased with formulation identity "
            f"(R-squared > {THRESHOLDS['formulation_alias_r2']:.2f} on formulation alone): "
            + ", ".join(aliased_f[:10])
            + ("..." if len(aliased_f) > 10 else "")
            + ". Each is a formulation label in disguise and cannot be credited "
            "independently of everything else that changed at the same time."
        )
    within_estimable = sorted(alias_f[alias_f <= THRESHOLDS["formulation_alias_r2"]].index)

    # aliasing with time
    if "manufacturing_date" in ds.metadata.columns:
        dates = ds.metadata.loc[ds.wide.index, "manufacturing_date"]
        alias_t = pd.Series({c: r2_on_time(Xl[c], dates) for c in live}, dtype=float)
        aliased_t = sorted(alias_t[alias_t > THRESHOLDS["time_alias_r2"]].index)
        if aliased_t:
            statements.append(
                f"{len(aliased_t)} predictor(s) track manufacturing date almost perfectly: "
                + ", ".join(aliased_t[:8])
                + ". A factor that only changed when the process changed is confounded with "
                "everything else that changed then, including anything unrecorded."
            )
    else:
        alias_t = pd.Series(np.nan, index=live, dtype=float)
        aliased_t = []
        notes.append("No manufacturing_date in metadata; time aliasing was not assessed.")

    # -- 3. mixture constraint ----------------------------------------
    mixture_groups, lin_dep = detect_mixture_constraints(ds, live)
    for g in mixture_groups:
        statements.append(
            f"Mixture constraint detected: {', '.join(g['columns'])} sum to a constant "
            f"({g['sum_mean']:.2f}, sd {g['sum_sd']:.4f}). Ordinary regression on these raw "
            "components is singular. Fit a mixture model, or omit one component as the "
            "dependent remainder; the app will not fit them all together."
        )
    for d in lin_dep:
        statements.append(
            "Exact linear dependency among predictors: "
            + " ".join(
                f"{coef:+.2f}*{col}" for col, coef in zip(d["columns"], d["coefficients"])
            )
            + " is constant. These cannot all appear in the same model."
        )

    # -- closing summary ----------------------------------------------
    answerable = [c for c in within_estimable if c not in zero_info]
    statements.append(
        "What this dataset can address: factors that vary within formulations, where "
        f"between-batch replication exists. There are {len(answerable)} of these."
    )
    statements.append(
        "What it cannot address: the individual contribution of any factor listed above as "
        "aliased or inseparable. No amount of modelling recovers information the data does "
        "not contain; only a designed experiment does."
    )

    return DesignAssessment(
        n_batches=n,
        n_formulations=ds.n_formulations,
        n_predictors=p,
        p_over_n_flag=bool(p_flag),
        correlation=corr,
        clusters=clusters,
        cluster_of=cluster_of,
        inseparable_groups=inseparable,
        vif=vif,
        formulation_alias_r2=alias_f,
        time_alias_r2=alias_t,
        aliased_with_formulation=aliased_f,
        aliased_with_time=aliased_t,
        mixture_groups=mixture_groups,
        linear_dependencies=lin_dep,
        range_table=range_table,
        zero_information=sorted(set(zero_info) | set(ds.excluded)),
        within_formulation_estimable=within_estimable,
        statements=statements,
        notes=notes,
    )


def detect_mixture_constraints(
    ds: BatchDataset, columns: List[str], tol: float = THRESHOLDS["mixture_sum_tolerance"]
) -> Tuple[List[Dict], List[Dict]]:
    """Find component sets that sum to a constant, plus general linear dependencies.

    Two mechanisms:
      1. FORMULATION columns expressed as percentages: test whether they sum to a
         constant (100 or otherwise), allowing for a fixed omitted remainder.
      2. SVD of the centred FORMULATION block: any near-zero singular value is an
         exact linear dependency, which covers mg/% duplication as well as mixtures.
    """
    form_cols = [c for c in columns if ds.families.get(c) == "FORMULATION"]
    X = ds.wide[form_cols].apply(pd.to_numeric, errors="coerce") if form_cols else pd.DataFrame()
    mixture: List[Dict] = []
    deps: List[Dict] = []
    if X.empty or X.shape[1] < 2:
        return mixture, deps

    pct_cols = [c for c in form_cols if X[c].notna().all() and _looks_like_percent(c)]
    if len(pct_cols) >= 2:
        # greedy: start from all percent columns, drop the one that most reduces the
        # sum's variability, until the sum is constant or fewer than 2 remain
        current = list(pct_cols)
        while len(current) >= 2:
            s = X[current].sum(axis=1)
            if float(s.std(ddof=0)) <= tol:
                mixture.append(
                    {
                        "columns": sorted(current),
                        "sum_mean": float(s.mean()),
                        "sum_sd": float(s.std(ddof=0)),
                    }
                )
                break
            # try removing each column, keep the removal with the smallest resulting sd
            best, best_sd = None, np.inf
            for c in current:
                trial = [x for x in current if x != c]
                sd = float(X[trial].sum(axis=1).std(ddof=0))
                if sd < best_sd:
                    best, best_sd = c, sd
            if best is None or best_sd >= float(s.std(ddof=0)):
                break
            current = [x for x in current if x != best]

    # SVD-based exact dependencies on the centred block
    A = X.dropna(axis=1, how="any")
    if A.shape[1] >= 2 and A.shape[0] >= 2:
        M = A.to_numpy(dtype=float)
        M = M - M.mean(axis=0)
        scale = M.std(axis=0, ddof=0)
        keep = scale > 0
        if keep.sum() >= 2:
            Ms = M[:, keep] / scale[keep]
            cols = list(np.array(A.columns)[keep])
            _, sv, vt = np.linalg.svd(Ms, full_matrices=False)
            for k, s in enumerate(sv):
                if s < 1e-8 * max(sv[0], 1e-12):
                    v = vt[k]
                    idx = np.argsort(-np.abs(v))
                    members = [cols[i] for i in idx if abs(v[i]) > 1e-6]
                    coefs = [float(v[i] / scale[keep][i]) for i in idx if abs(v[i]) > 1e-6]
                    if len(members) >= 2:
                        deps.append({"columns": members, "coefficients": coefs})
    return mixture, deps


def _looks_like_percent(name: str) -> bool:
    low = name.lower()
    return ("pct" in low) or ("%" in low) or low.endswith("_ww") or ("percent" in low)


def mixture_safe_predictors(ds: BatchDataset, design: DesignAssessment, columns: List[str]) -> Tuple[List[str], List[str]]:
    """Drop one component per mixture group as the dependent remainder.

    Returns (kept, removed_with_reason_applied). The removed column is the one with
    the largest variance, so the retained set keeps the most interpretable members.
    """
    kept = list(columns)
    removed: List[str] = []
    for g in design.mixture_groups:
        members = [c for c in g["columns"] if c in kept]
        if len(members) < 2:
            continue
        sds = ds.wide[members].apply(pd.to_numeric, errors="coerce").std(ddof=0)
        drop = str(sds.idxmax())
        kept.remove(drop)
        removed.append(drop)
    for d in design.linear_dependencies:
        members = [c for c in d["columns"] if c in kept]
        if len(members) >= 2:
            drop = members[-1]
            kept.remove(drop)
            removed.append(drop)
    return kept, removed
