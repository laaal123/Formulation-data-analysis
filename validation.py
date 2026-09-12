"""Prompt 5 -- validation.

Grouped, nested, permuted. The permutation test decides whether anything
downstream is allowed to be displayed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, LeaveOneGroupOut

from config import RANDOM_SEED, THRESHOLDS, RunSettings, library_versions
from errors import Refusal
from modelling import AttributionModel, ClusterPlan, StabilityResult, stability_selection

CV_SCHEMES = ("leave_one_formulation_out", "grouped_kfold_formulation", "leave_one_batch_out")


def q_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Cross-validated R^2 computed against the mean of the observed data (Q^2).
    Can be negative; a negative value means the model is worse than the mean."""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    if m.sum() < 2:
        return np.nan
    ss_res = float(((y_true[m] - y_pred[m]) ** 2).sum())
    ss_tot = float(((y_true[m] - y_true[m].mean()) ** 2).sum())
    if ss_tot == 0:
        return np.nan
    return 1.0 - ss_res / ss_tot


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    m = np.isfinite(y_true) & np.isfinite(y_pred)
    return float(np.sqrt(np.mean((y_true[m] - y_pred[m]) ** 2))) if m.sum() else np.nan


# ---------------------------------------------------------------------
# splitting
# ---------------------------------------------------------------------
def make_splits(groups: pd.Series, scheme: str, n_splits: int = 5) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Grouped splits only. There is no random row split available anywhere in
    this module, by design: batches of one formulation share nearly all predictor
    values, so a random split leaks the answer into the test set."""
    if scheme not in CV_SCHEMES:
        raise ValueError(f"scheme must be one of {CV_SCHEMES}")
    n = len(groups)
    g = groups.to_numpy()
    if scheme == "leave_one_batch_out":
        return [(np.delete(np.arange(n), i), np.array([i])) for i in range(n)]
    n_groups = pd.Series(g).nunique()
    if n_groups < 2:
        raise Refusal(
            "Only one formulation is present, so a formulation cannot be held out. "
            "This dataset cannot validate any formulation-level attribution."
        )
    if scheme == "leave_one_formulation_out":
        return list(LeaveOneGroupOut().split(np.zeros(n), groups=g))
    k = int(min(n_splits, n_groups))
    return list(GroupKFold(n_splits=k).split(np.zeros(n), groups=g))


def cross_validate(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    plan: ClusterPlan,
    model_type: str = "elastic_net",
    scheme: str = "leave_one_formulation_out",
    seed: int = RANDOM_SEED,
    inner_cv_splits: int = 5,
) -> Dict:
    """Outer loop reports; the inner loop tunes. Tuning never sees the outer test fold."""
    y = pd.to_numeric(y, errors="coerce")
    keep = y.notna()
    X, y, groups = X.loc[keep], y.loc[keep], groups.loc[keep]
    if y.nunique() < 2:
        raise Refusal(
            "The response takes a single value across every batch with data. There is no "
            "variation to predict, so cross-validation would be meaningless."
        )
    if len(y) < 4:
        raise Refusal(
            f"{len(y)} batch(es) have this response. Cross-validation needs at least four."
        )
    splits = make_splits(groups, scheme)
    pred = pd.Series(np.nan, index=y.index, dtype=float)
    fold_id = pd.Series(-1, index=y.index, dtype=int)
    per_fold = []
    selected_per_fold = []
    for f, (tr, te) in enumerate(splits):
        ytr = y.iloc[tr]
        if ytr.nunique() < 2 or len(tr) < 4:
            continue
        model = AttributionModel(model_type, plan, seed=seed, inner_cv_splits=inner_cv_splits)
        model.fit(X.iloc[tr], ytr, groups.iloc[tr])
        p = model.predict(X.iloc[te])
        pred.iloc[te] = p
        fold_id.iloc[te] = f
        per_fold.append(
            {
                "fold": f,
                "held_out": sorted(map(str, pd.unique(groups.iloc[te]))),
                "n_test": len(te),
                "rmse": rmse(y.iloc[te].to_numpy(), p),
                "params": dict(model.params_),
                "n_selected": len(model.selected_),
            }
        )
        selected_per_fold.append(list(model.selected_))
    if not per_fold:
        raise Refusal(
            "No cross-validation fold could be fitted: every training split was too small or "
            "had no variation in the response. This dataset cannot be validated."
        )
    return {
        "y_true": y,
        "y_pred": pred,
        "fold": fold_id,
        "q2": q_squared(y.to_numpy(), pred.to_numpy()),
        "rmse": rmse(y.to_numpy(), pred.to_numpy()),
        "per_fold": pd.DataFrame(per_fold),
        "selected_per_fold": selected_per_fold,
        "scheme": scheme,
        "n_used": int(pred.notna().sum()),
    }


def random_split_q2(
    X: pd.DataFrame,
    y: pd.Series,
    plan: ClusterPlan,
    model_type: str = "elastic_net",
    n_splits: int = 5,
    seed: int = RANDOM_SEED,
) -> float:
    """Deliberately wrong random-row CV, computed for one purpose only: to show the
    user how much optimism a random split buys. Never used for reporting."""
    from sklearn.model_selection import KFold

    y = pd.to_numeric(y, errors="coerce")
    keep = y.notna()
    X, y = X.loc[keep], y.loc[keep]
    pred = pd.Series(np.nan, index=y.index, dtype=float)
    kf = KFold(n_splits=min(n_splits, len(y)), shuffle=True, random_state=seed)
    for tr, te in kf.split(np.zeros(len(y))):
        if y.iloc[tr].nunique() < 2:
            continue
        m = AttributionModel(model_type, plan, seed=seed).fit(X.iloc[tr], y.iloc[tr])
        pred.iloc[te] = m.predict(X.iloc[te])
    return q_squared(y.to_numpy(), pred.to_numpy())


# ---------------------------------------------------------------------
# permutation test
# ---------------------------------------------------------------------
def permutation_test(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    plan: ClusterPlan,
    model_type: str = "elastic_net",
    scheme: str = "leave_one_formulation_out",
    n_permutations: int = 1000,
    seed: int = RANDOM_SEED,
    observed_q2: Optional[float] = None,
    inner_cv_splits: int = 5,
    n_jobs: int = 1,
) -> Dict:
    """Shuffle the response, refit the ENTIRE pipeline including tuning and variable
    selection, and rebuild the null distribution of cross-validated Q^2."""
    if observed_q2 is None:
        observed_q2 = cross_validate(
            X, y, groups, plan, model_type, scheme, seed, inner_cv_splits
        )["q2"]
    y = pd.to_numeric(y, errors="coerce")
    keep = y.notna()
    Xk, yk, gk = X.loc[keep], y.loc[keep], groups.loc[keep]
    rng = np.random.default_rng(seed + 1)
    perms = [rng.permutation(len(yk)) for _ in range(n_permutations)]

    def one(perm):
        ys = pd.Series(yk.to_numpy()[perm], index=yk.index)
        try:
            return cross_validate(Xk, ys, gk, plan, model_type, scheme, seed, inner_cv_splits)["q2"]
        except Exception:  # noqa: BLE001
            return np.nan

    if n_jobs != 1:
        from joblib import Parallel, delayed

        null = Parallel(n_jobs=n_jobs, backend="loky")(delayed(one)(p) for p in perms)
    else:
        null = [one(p) for p in perms]
    null = np.array([v for v in null if np.isfinite(v)], dtype=float)
    if null.size == 0 or not np.isfinite(observed_q2):
        return {
            "observed_q2": observed_q2,
            "null": null,
            "p_value": np.nan,
            "null_p95": np.nan,
            "n_permutations": int(null.size),
            "passed": False,
            "note": "permutation null could not be built; treat as not supported",
        }
    p_value = float((1 + np.sum(null >= observed_q2)) / (null.size + 1))
    return {
        "observed_q2": float(observed_q2),
        "null": null,
        "p_value": p_value,
        "null_p95": float(np.percentile(null, 95)),
        "null_median": float(np.median(null)),
        "n_permutations": int(null.size),
        "passed": bool(p_value <= THRESHOLDS["permutation_alpha"]),
        "note": "",
    }


# ---------------------------------------------------------------------
# full validation run
# ---------------------------------------------------------------------
@dataclass
class ValidationResult:
    cv: Dict
    permutation: Dict
    stability: StabilityResult
    verdict: str                       # supported | weak | not_supported
    verdict_text: str
    settings: Dict
    random_split_q2: Optional[float] = None
    statements: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        """Gate for everything in the Findings tab."""
        return self.verdict in ("supported", "weak") and bool(self.permutation.get("passed"))

    def observed_vs_predicted(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "observed": self.cv["y_true"],
                "predicted_heldout": self.cv["y_pred"],
                "fold": self.cv["fold"],
            }
        )

    def text(self) -> str:
        return "\n".join(self.statements)


def run_validation(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    plan: ClusterPlan,
    model_type: str = "elastic_net",
    scheme: str = "leave_one_formulation_out",
    settings: Optional[RunSettings] = None,
    compare_random_split: bool = True,
    n_jobs: int = 1,
) -> ValidationResult:
    st = settings or RunSettings()
    warnings: List[str] = []

    n_groups = int(pd.Series(groups[pd.to_numeric(y, errors="coerce").notna()]).nunique())
    if scheme.startswith("leave_one_formulation") and n_groups < 3:
        warnings.append(
            f"Only {n_groups} formulation(s) with data: leave-one-formulation-out gives "
            f"{n_groups} folds, each training on very little. Results are indicative at best."
        )
    cv = cross_validate(X, y, groups, plan, model_type, scheme, st.seed, st.inner_cv_splits)
    perm = permutation_test(
        X, y, groups, plan, model_type, scheme, st.n_permutations, st.seed,
        observed_q2=cv["q2"], inner_cv_splits=st.inner_cv_splits, n_jobs=n_jobs,
    )
    stab = stability_selection(
        X.loc[pd.to_numeric(y, errors="coerce").notna()],
        pd.to_numeric(y, errors="coerce").dropna(),
        groups.loc[pd.to_numeric(y, errors="coerce").notna()],
        plan, model_type, st.n_bootstrap, st.seed, inner_cv_splits=st.inner_cv_splits,
    )
    rnd = None
    if compare_random_split:
        try:
            rnd = random_split_q2(X, y, plan, model_type, seed=st.seed)
        except Exception:  # noqa: BLE001
            rnd = None

    q2 = cv["q2"]
    stable = stab.frequency[stab.frequency >= st.stability_threshold]
    if not perm.get("passed"):
        verdict = "not_supported"
    elif np.isfinite(q2) and q2 >= 0.4 and len(stable) >= 1:
        verdict = "supported"
    else:
        verdict = "weak"

    statements: List[str] = []
    statements.append(
        f"Cross-validation: {scheme}, {cv['n_used']} batches predicted from folds they were "
        f"not fitted on. Q-squared = {q2:.3f}, RMSE = {cv['rmse']:.3g}."
    )
    statements.append(
        f"Permutation test: {perm['n_permutations']} shuffles of the response, whole pipeline "
        f"refitted each time including tuning and variable selection. "
        f"Observed Q-squared {perm['observed_q2']:.3f} against a null whose 95th percentile is "
        f"{perm['null_p95']:.3f} (p = {perm['p_value']:.3f})."
        if np.isfinite(perm.get("p_value", np.nan))
        else "Permutation test could not be completed; treat the result as unsupported."
    )
    statements.append(
        f"Stability selection: {stab.n_effective} usable resamples of {stab.n_boot}; "
        f"{len(stable)} feature(s) selected in at least "
        f"{st.stability_threshold:.0%} of them."
    )
    if rnd is not None and np.isfinite(rnd) and np.isfinite(q2):
        statements.append(
            f"For comparison only: a random row split on the same data gives Q-squared "
            f"{rnd:.3f} against the grouped {q2:.3f}. The grouped number is the one reported; "
            "the difference is the optimism a random split would have bought you."
        )

    if verdict == "not_supported":
        verdict_text = (
            "NOT SUPPORTED BY THIS DATA. The model does not beat its own permutation null, "
            "so no attribution can be made and no importance ranking will be displayed. "
            "Any apparent relationship in the raw data does not survive held-out validation."
        )
    elif verdict == "supported":
        verdict_text = (
            "SUPPORTED as an association. The model beats its permutation null and predicts "
            "held-out formulations, and at least one factor is stably selected. This is still "
            "an association from observational batch data, and every factor carries the "
            "confound list from the design diagnostics."
        )
    else:
        verdict_text = (
            "WEAK - NEEDS CONFIRMATION. The model beats its permutation null, but held-out "
            "prediction is poor or no factor is stably selected. Treat the ranking as a "
            "hypothesis to test, not a result."
        )
    statements.append(verdict_text)

    return ValidationResult(
        cv=cv,
        permutation=perm,
        stability=stab,
        verdict=verdict,
        verdict_text=verdict_text,
        settings={**st.to_dict(), "model_type": model_type, "scheme": scheme,
                  "libraries": library_versions()},
        random_split_q2=rnd,
        statements=statements,
        warnings=warnings,
    )
