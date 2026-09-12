"""Prompt 4 -- modelling under p >> n.

The model is a single refittable object. Everything data-dependent -- cluster
construction, standardisation, hyperparameter choice, variable selection --
happens inside fit(), so that cross-validation, permutation and bootstrap loops
refit all of it and none of it leaks.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.cross_decomposition import PLSRegression
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import ElasticNet, Ridge, enet_path
from sklearn.model_selection import GroupKFold, KFold

LINEAR_TYPES = ("elastic_net", "lasso", "ridge")
ALPHA_GRID = np.logspace(-3.0, 1.5, 25)[::-1]        # descending, as paths require
RIDGE_ALPHA_GRID = np.logspace(-2.0, 3.0, 25)[::-1]


class _LinearFit:
    """Coefficients from a regularisation path, in the same shape sklearn uses."""

    def __init__(self, coef: np.ndarray, intercept: float):
        self.coef_ = np.asarray(coef, dtype=float).ravel()
        self.intercept_ = float(intercept)

    def predict(self, Z: np.ndarray) -> np.ndarray:
        return np.asarray(Z, dtype=float) @ self.coef_ + self.intercept_


def _enet_path_coefs(Z: np.ndarray, y: np.ndarray, l1_ratio: float, alphas: np.ndarray):
    """Coefficients and intercepts for a whole alpha grid in one pass."""
    zm = Z.mean(axis=0)
    ym = float(y.mean())
    Zc = Z - zm
    yc = y - ym
    if l1_ratio >= 1.0:
        l1_ratio = 1.0
    _, coefs, _ = enet_path(
        np.asfortranarray(Zc), yc, l1_ratio=max(l1_ratio, 1e-3), alphas=alphas,
        max_iter=20000, tol=1e-4, precompute=False, check_input=False,
    )
    coefs = np.asarray(coefs)                     # (p, n_alphas)
    intercepts = ym - zm @ coefs                  # (n_alphas,)
    return coefs, intercepts


def _orthonormalise_block(Zg: np.ndarray, tol: float = 1e-8):
    """Return (Q, back) with (1/n) Q'Q = I, and a map from block coefficients to member
    coefficients. Rank-deficient blocks keep only their real directions, so a cluster of
    six near-identical columns is treated as the one direction it actually is."""
    n = Zg.shape[0]
    U, d, Vt = np.linalg.svd(Zg, full_matrices=False)
    keep = d > tol * max(d[0] if d.size else 0.0, 1e-12)
    if not keep.any():
        return np.zeros((n, 0)), np.zeros((Zg.shape[1], 0))
    U, d, Vt = U[:, keep], d[keep], Vt[keep]
    Q = U * np.sqrt(n)
    back = (Vt.T * (1.0 / d)) * np.sqrt(n)      # b_members = back @ b_block
    return Q, back


def group_lasso_path(
    Z: np.ndarray,
    y: np.ndarray,
    blocks: List[np.ndarray],
    lambdas: np.ndarray,
    max_iter: int = 50,
    tol: float = 1e-6,
):
    """Block coordinate descent for the group lasso.

        min (1/2n)||y - Zb||^2 + lambda * sum_g sqrt(p_g) ||b_g||_2

    A whole correlated group enters or leaves together, which is the point: the app
    should never report one member of an inseparable cluster and drop its twin.
    Blocks are orthonormalised, so each block update is closed form.

    Covariance updates plus an active set: residual correlations are maintained through
    a Gram matrix instead of touching the data each time, and a group at zero is only
    revisited to check its optimality condition. Same solution, far less work.
    """
    n = Z.shape[0]
    ym = float(y.mean())
    yc = y - ym

    cols, slices, backs, weights = [], [], [], []
    cursor = 0
    for idx in blocks:
        Q, back = _orthonormalise_block(Z[:, idx])
        r = Q.shape[1]
        if r == 0:
            slices.append(slice(cursor, cursor))
            backs.append(back)
            weights.append(1.0)
            continue
        cols.append(Q)
        slices.append(slice(cursor, cursor + r))
        backs.append(back)
        weights.append(np.sqrt(r))
        cursor += r
    if not cols:
        return np.zeros((Z.shape[1], len(lambdas))), np.full(len(lambdas), ym)

    Q = np.hstack(cols)
    G = (Q.T @ Q) / n
    Qty = (Q.T @ yc) / n
    m = Q.shape[1]
    b = np.zeros(m)
    c = Qty.copy()                      # c = (1/n) Q'(y - Q b)
    live = [g for g, sl in enumerate(slices) if sl.stop > sl.start]

    coefs = np.zeros((Z.shape[1], len(lambdas)))
    intercepts = np.full(len(lambdas), ym)

    def update(g, lam) -> float:
        sl = slices[g]
        thresh = lam * weights[g]
        if sl.stop - sl.start == 1:
            # singleton: soft thresholding on a scalar. Most blocks are singletons and
            # the vector path's allocations dominated the solve time.
            i = sl.start
            bi = b[i]
            s = c[i] + bi
            new = s - thresh if s > thresh else (s + thresh if s < -thresh else 0.0)
            delta = new - bi
            if delta != 0.0:
                b[i] = new
                c[:] -= G[:, i] * delta
                return abs(delta)
            return 0.0
        bg = b[sl]
        s = c[sl] + bg
        norm = float(np.linalg.norm(s))
        new = (1.0 - thresh / norm) * s if norm > thresh else np.zeros_like(s)
        delta = new - bg
        if np.any(delta):
            b[sl] = new
            c[:] -= G[:, sl] @ delta
            return float(np.max(np.abs(delta)))
        return 0.0

    for k, lam in enumerate(lambdas):
        for _outer in range(max_iter):
            active = [g for g in live if np.any(b[slices[g]] != 0.0)]
            for _inner in range(max_iter):
                moved = max((update(g, lam) for g in active), default=0.0)
                if moved < tol:
                    break
            # optimality check on everything currently at zero
            added = False
            for g in live:
                if np.any(b[slices[g]] != 0.0):
                    continue
                sl = slices[g]
                viol = (
                    abs(c[sl.start]) if sl.stop - sl.start == 1
                    else float(np.linalg.norm(c[sl]))
                )
                if viol > lam * weights[g] + tol:
                    update(g, lam)
                    added = True
            if not added:
                break
        out = np.zeros(Z.shape[1])
        for g, idx in enumerate(blocks):
            sl = slices[g]
            if sl.stop > sl.start:
                out[idx] = backs[g] @ b[sl]
        coefs[:, k] = out
    return coefs, intercepts


def group_lasso_lambda_max(Z: np.ndarray, y: np.ndarray, blocks: List[np.ndarray]) -> float:
    """Smallest penalty at which every group is zero."""
    n = Z.shape[0]
    yc = y - y.mean()
    best = 0.0
    for idx in blocks:
        Q, _ = _orthonormalise_block(Z[:, idx])
        if Q.shape[1] == 0:
            continue
        val = float(np.linalg.norm(Q.T @ yc) / n / np.sqrt(Q.shape[1]))
        best = max(best, val)
    return max(best, 1e-6)


# ---------------------------------------------------------------------
# Scheffe canonical mixture polynomials
# ---------------------------------------------------------------------
def pseudocomponents(X: pd.DataFrame, components: Sequence[str]) -> np.ndarray:
    """Rescale components so each row sums to one.

    Mixture components sum to a constant, so their effects are only ever relative.
    The Scheffe canonical form handles that by dropping the intercept rather than by
    dropping a component, which is why it says something the ordinary model cannot.
    """
    M = X.reindex(columns=list(components)).apply(pd.to_numeric, errors="coerce").to_numpy(float)
    total = np.nansum(M, axis=1, keepdims=True)
    total[total == 0] = np.nan
    return M / total


def scheffe_terms(P: np.ndarray, components: Sequence[str], degree: int):
    """Design matrix and term names for a Scheffe polynomial of degree 1 or 2."""
    names = list(components)
    cols = [P[:, i] for i in range(P.shape[1])]
    if degree >= 2:
        for i in range(P.shape[1]):
            for j in range(i + 1, P.shape[1]):
                cols.append(P[:, i] * P[:, j])
                names.append(f"{components[i]} x {components[j]}")
    return np.column_stack(cols), names


def _ridge_path_coefs(Z: np.ndarray, y: np.ndarray, alphas: np.ndarray):
    """Closed-form ridge for every alpha via one SVD."""
    zm = Z.mean(axis=0)
    ym = float(y.mean())
    Zc = Z - zm
    U, d, Vt = np.linalg.svd(Zc, full_matrices=False)
    uty = U.T @ (y - ym)
    coefs = np.empty((Z.shape[1], len(alphas)))
    for k, a in enumerate(alphas):
        shrink = d / (d**2 + a)
        coefs[:, k] = Vt.T @ (shrink * uty)
    intercepts = ym - zm @ coefs
    return coefs, intercepts

from config import RANDOM_SEED, THRESHOLDS
from errors import Refusal

MODEL_TYPES = (
    "elastic_net", "lasso", "group_lasso", "ridge", "pls",
    "scheffe_linear", "scheffe_quadratic", "gbm", "rf",
)
# models whose zero coefficients mean "not selected"
SPARSE_TYPES = ("elastic_net", "lasso", "group_lasso")
# models that keep every predictor; for these, "selected" is defined as being inside
# the complexity cap by absolute effect, otherwise stability selection would report
# 100% for everything and mean nothing
DENSE_TYPES = ("ridge", "pls", "scheffe_linear", "scheffe_quadratic", "gbm", "rf")
MIXTURE_TYPES = ("scheffe_linear", "scheffe_quadratic")


# ---------------------------------------------------------------------
# 1. univariate screen
# ---------------------------------------------------------------------
def benjamini_hochberg(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    ok = np.isfinite(p)
    q = np.full(p.shape, np.nan)
    if ok.sum() == 0:
        return q
    pv = p[ok]
    m = pv.size
    order = np.argsort(pv)
    ranked = pv[order]
    adj = ranked * m / np.arange(1, m + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0, 1)
    out = np.empty(m)
    out[order] = adj
    q[ok] = out
    return q


def univariate_screen(X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Each predictor alone against the response. A SCREEN, never a conclusion."""
    rows = []
    yv = pd.to_numeric(y, errors="coerce")
    for c in X.columns:
        x = pd.to_numeric(X[c], errors="coerce")
        m = x.notna() & yv.notna()
        if m.sum() < 3 or x[m].nunique() < 2 or yv[m].nunique() < 2:
            rows.append({"predictor": c, "r": np.nan, "p_value": np.nan, "n": int(m.sum())})
            continue
        r, p = stats.pearsonr(x[m], yv[m])
        rows.append({"predictor": c, "r": float(r), "p_value": float(p), "n": int(m.sum())})
    out = pd.DataFrame(rows)
    out["q_value_bh"] = benjamini_hochberg(out["p_value"].to_numpy())
    out = out.sort_values("p_value", na_position="last").reset_index(drop=True)
    out.attrs["caption"] = (
        "Univariate screen only. These p-values ignore every other predictor and the "
        "confounding between them; they are a filter for what to look at next, not findings."
    )
    return out


# ---------------------------------------------------------------------
# complexity cap
# ---------------------------------------------------------------------
def complexity_cap(n_train: int) -> Dict[str, object]:
    """One retained predictor per five batches, at least one, at most ten."""
    k = int(max(1, min(10, np.floor(n_train / 5))))
    return {
        "max_features": k,
        "rule": f"max_features = clip(floor(n_train/5), 1, 10) = {k} for n_train={n_train}",
    }


# ---------------------------------------------------------------------
# the model
# ---------------------------------------------------------------------
@dataclass
class ClusterPlan:
    """How raw predictors map onto model features. Fixed before fitting, from the
    design diagnostics, so it is identical in every resample."""

    groups: Dict[str, List[str]] = field(default_factory=dict)   # feature name -> members
    dropped: Dict[str, str] = field(default_factory=dict)        # column -> reason
    # every mixture component, INCLUDING the one dropped as the remainder elsewhere:
    # a Scheffe model needs the complete simplex and handles the constraint by having
    # no intercept
    mixture_components: List[str] = field(default_factory=list)

    @classmethod
    def from_design(cls, design, columns: Sequence[str], mixture_drop: Sequence[str] = ()) -> "ClusterPlan":
        cols = [c for c in columns if c not in set(mixture_drop)]
        groups: Dict[str, List[str]] = {}
        seen = set()
        for cluster in design.clusters:
            members = [m for m in cluster if m in cols]
            if not members:
                continue
            name = members[0] if len(members) == 1 else f"cluster[{members[0]}+{len(members)-1}]"
            groups[name] = members
            seen.update(members)
        for c in cols:
            if c not in seen:
                groups[c] = [c]
        dropped = {c: "removed to break a mixture / linear dependency" for c in mixture_drop}
        mixture = sorted({c for g in design.mixture_groups for c in g["columns"]})
        return cls(groups=groups, dropped=dropped, mixture_components=mixture)

    @classmethod
    def identity(cls, columns: Sequence[str]) -> "ClusterPlan":
        return cls(groups={c: [c] for c in columns})

    @property
    def feature_names(self) -> List[str]:
        return list(self.groups.keys())

    def is_cluster(self, feature: str) -> bool:
        return len(self.groups.get(feature, [])) > 1


class AttributionModel:
    """Refittable end-to-end model: cluster scores -> standardise -> penalised fit.

    Nothing here is fitted on anything but the rows handed to fit().
    """

    def __init__(
        self,
        model_type: str = "elastic_net",
        plan: Optional[ClusterPlan] = None,
        seed: int = RANDOM_SEED,
        inner_cv_splits: int = 5,
        max_features: Optional[int] = None,
        enforce_cap: bool = True,
    ):
        if model_type not in MODEL_TYPES:
            raise ValueError(f"model_type must be one of {MODEL_TYPES}")
        self.model_type = model_type
        self.plan = plan
        self.seed = seed
        self.inner_cv_splits = inner_cv_splits
        self.max_features = max_features
        self.enforce_cap = enforce_cap
        self.fitted_ = False

    # -- feature construction -----------------------------------------
    def _layout(self, X: pd.DataFrame) -> None:
        """Fix the raw-column order and the group membership matrix once per fit."""
        raw: List[str] = []
        feats: List[str] = []
        members_by_feat: List[List[str]] = []
        for name, members in self.plan.groups.items():
            present = [m for m in members if m in X.columns]
            if not present:
                continue
            feats.append(name)
            members_by_feat.append(present)
            raw.extend(present)
        raw = list(dict.fromkeys(raw))
        pos = {c: i for i, c in enumerate(raw)}
        A = np.zeros((len(raw), len(feats)))
        for j, members in enumerate(members_by_feat):
            for m in members:
                A[pos[m], j] = 1.0
        self._raw_cols_ = raw
        self._layout_features_ = feats
        self._members_by_feature_ = members_by_feat
        self._A_ = A

    def _raw_matrix(self, X: pd.DataFrame) -> np.ndarray:
        block = X.reindex(columns=self._raw_cols_)
        return block.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)

    def _build_features(self, X: pd.DataFrame, train: bool) -> pd.DataFrame:
        M = self._raw_matrix(X)
        if train:
            with np.errstate(invalid="ignore"):
                mu = np.nanmean(np.where(np.isfinite(M), M, np.nan), axis=0)
                sd = np.nanstd(np.where(np.isfinite(M), M, np.nan), axis=0, ddof=0)
            mu = np.where(np.isfinite(mu), mu, 0.0)
            sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nan)
            self._mu_, self._sd_ = mu, sd
            # sign alignment inside a cluster: members correlated negatively with the
            # reference are flipped before averaging, so a cluster score is coherent
            signs = np.ones(M.shape[1])
            pos = {c: i for i, c in enumerate(self._raw_cols_)}
            for members in self._members_by_feature_:
                if len(members) < 2:
                    continue
                ref = M[:, pos[members[0]]]
                for m in members[1:]:
                    col = M[:, pos[m]]
                    msk = np.isfinite(ref) & np.isfinite(col)
                    if msk.sum() > 2 and np.std(ref[msk]) > 0 and np.std(col[msk]) > 0:
                        r = float(np.corrcoef(ref[msk], col[msk])[0, 1])
                        if np.isfinite(r) and r < 0:
                            signs[pos[m]] = -1.0
            self._signs_ = signs
        Z = (M - self._mu_) / self._sd_ * self._signs_
        finite = np.isfinite(Z)
        S = np.where(finite, Z, 0.0) @ self._A_
        C = finite.astype(float) @ self._A_
        with np.errstate(invalid="ignore", divide="ignore"):
            Fv = np.where(C > 0, S / np.where(C > 0, C, 1.0), np.nan)
        F = pd.DataFrame(Fv, index=X.index, columns=self._layout_features_)
        if train:
            # a cluster whose members were all constant in this split carries nothing
            self._dead_features_ = [
                c for c in F.columns
                if F[c].isna().all() or not np.isfinite(F[c].std(ddof=0)) or F[c].std(ddof=0) == 0
            ]
        F = F.drop(columns=[c for c in self._dead_features_ if c in F.columns])
        n_filled = int(F.isna().to_numpy().sum())
        if train:
            self.filled_cells_ = n_filled
            self.filled_features_ = [c for c in F.columns if F[c].isna().any()]
        else:
            self.filled_cells_predict_ = n_filled
        # a missing feature falls back to the training mean (z = 0). This is an
        # imputation: it is counted and surfaced, never silent.
        return F.fillna(0.0)

    # -- fit -----------------------------------------------------------
    def fit(self, X: pd.DataFrame, y: pd.Series, groups: Optional[pd.Series] = None) -> "AttributionModel":
        if self.plan is None:
            self.plan = ClusterPlan.identity(list(X.columns))
        self._dead_features_: List[str] = []
        self._layout(X)

        y = pd.to_numeric(y, errors="coerce")
        mask = y.notna().to_numpy()          # positional: resampled frames repeat labels
        X, y = X.iloc[mask], y.iloc[mask]
        groups = groups.iloc[mask] if groups is not None else None
        n = len(y)
        if n < 3:
            raise Refusal(
                f"Refusing to fit on {n} observations: no model of any kind is "
                "identifiable, and cross-validation is impossible."
            )

        F = self._build_features(X, train=True)
        self.feature_names_ = list(F.columns)
        self.n_train_ = n
        cap = complexity_cap(n)
        self.cap_ = cap if self.max_features is None else {
            "max_features": int(self.max_features),
            "rule": f"max_features set explicitly to {self.max_features}",
        }
        self.y_mean_ = float(y.mean())
        self.y_sd_ = float(y.std(ddof=0)) or 1.0

        Fx = F.to_numpy(dtype=float)
        self.x_mean_ = Fx.mean(axis=0)
        sd = Fx.std(axis=0, ddof=0)
        sd[sd == 0] = 1.0
        self.x_sd_ = sd
        Z = (Fx - self.x_mean_) / self.x_sd_
        yv = y.to_numpy(dtype=float)

        if self.model_type in MIXTURE_TYPES:
            self._fit_scheffe(X, yv, groups)
        elif self.model_type == "group_lasso":
            self._fit_group_lasso(X, yv, groups)
        elif self.model_type in LINEAR_TYPES:
            self._fit_linear(Z, yv, groups)
        else:
            self.params_ = self._tune(Z, yv, groups)
            self._fit_final(Z, yv)
        self._apply_dense_selection()
        self._set_feature_members()
        self.fitted_ = True
        return self

    def _set_feature_members(self) -> None:
        if self.model_type == "group_lasso":
            self.feature_members_ = {
                f: list(m) for f, m in zip(self.feature_names_, self._block_members_)
            }
        elif self.model_type in MIXTURE_TYPES:
            self.feature_members_ = {
                f: [c for c in self._components_ if c in f.split(" x ")]
                or [c for c in self._components_ if c == f]
                for f in self.feature_names_
            }
        else:
            self.feature_members_ = {
                f: list(self.plan.groups.get(f, [f])) for f in self.feature_names_
            }

    def _apply_dense_selection(self) -> None:
        """For models that keep every predictor, define selection as being inside the
        complexity cap by absolute effect. Without this, stability selection reports
        100% for every feature of a ridge or PLS fit and carries no information."""
        if self.model_type in DENSE_TYPES and len(self.coef_):
            cap = self.cap_["max_features"]
            order = np.argsort(-np.abs(self.coef_))[:cap]
            self.selected_ = [self.feature_names_[i] for i in order if abs(self.coef_[i]) > 1e-12]
            self.selection_rule_ = f"top {cap} features by |effect| (model is not sparse)"
        else:
            self.selection_rule_ = "non-zero coefficient"

    # -- group lasso: whole clusters enter or leave together -------------
    def _member_matrix(self, X: pd.DataFrame, train: bool) -> np.ndarray:
        M = self._raw_matrix(X)
        if train:
            with np.errstate(invalid="ignore"):
                mu = np.nanmean(M, axis=0)
                sd = np.nanstd(M, axis=0, ddof=0)
            self._mu_ = np.where(np.isfinite(mu), mu, 0.0)
            sd = np.where(np.isfinite(sd) & (sd > 0), sd, np.nan)
            self._sd_ = sd
            self._signs_ = np.ones(M.shape[1])
            self._live_raw_ = np.isfinite(sd)
        Z = (M - self._mu_) / self._sd_
        Z = np.where(np.isfinite(Z), Z, 0.0)
        return Z

    def _fit_group_lasso(self, X: pd.DataFrame, y: np.ndarray, groups) -> None:
        Z = self._member_matrix(X, train=True)
        pos = {c: i for i, c in enumerate(self._raw_cols_)}
        blocks, names, members = [], [], []
        for feat, mem in zip(self._layout_features_, self._members_by_feature_):
            idx = np.array([pos[m] for m in mem if self._live_raw_[pos[m]]], dtype=int)
            if idx.size == 0:
                continue
            blocks.append(idx)
            names.append(feat)
            members.append([m for m in mem if self._live_raw_[pos[m]]])
        if not blocks:
            raise Refusal("No predictor varies in this training split; nothing to fit.")
        self.feature_names_ = names
        self._blocks_ = blocks
        self._block_members_ = members

        lam_max = group_lasso_lambda_max(Z, y, blocks)
        lambdas = np.logspace(np.log10(lam_max), np.log10(lam_max * 1e-3), 14)
        n = len(y)
        splits = self._splits(n, groups)
        sse = np.zeros(len(lambdas))
        used = 0
        for tr, te in splits:
            if len(tr) < 2 or len(te) < 1 or np.std(y[tr]) == 0:
                continue
            coefs, inter = group_lasso_path(Z[tr], y[tr], blocks, lambdas)
            pred = Z[te] @ coefs + inter
            sse += ((y[te][:, None] - pred) ** 2).sum(axis=0)
            used += len(te)
        k = int(np.argmin(sse / used)) if used else len(lambdas) - 1

        coefs, inter = group_lasso_path(Z, y, blocks, lambdas)
        cap = self.cap_["max_features"]

        def n_groups(col):
            return sum(1 for idx in blocks if np.any(np.abs(col[idx]) > 1e-10))

        capped = False
        while k > 0 and n_groups(coefs[:, k]) > cap:
            k -= 1                       # descending penalty grid: lower index = stronger
            capped = True
        b = coefs[:, k]
        self.params_ = {"lambda": float(lambdas[k]), "cap_applied": capped,
                        "penalty": "group lasso, sqrt(rank) weights"}
        self._member_coefs_ = {
            feat: {m: float(b[pos[m]]) for m in mem}
            for feat, mem in zip(names, members)
        }
        self._b_members_ = b
        self._intercept_ = float(inter[k])
        # a group's effect when its members move together, which is how they move
        self.coef_ = np.array([sum(self._member_coefs_[f].values()) for f in names])
        self.x_sd_ = np.ones(len(names))
        self.x_mean_ = np.zeros(len(names))
        self.selected_ = [f for f, c in zip(names, self.coef_) if abs(c) > 1e-10]
        self.estimator_ = None

    # -- Scheffe canonical mixture model ---------------------------------
    def _scheffe_setup(self, X: pd.DataFrame):
        comps = [c for c in (self.plan.mixture_components or []) if c in X.columns]
        if len(comps) < 2:
            raise Refusal(
                "A Scheffe mixture model needs at least two components that sum to a "
                "constant. None were detected in this dataset, so there is no simplex "
                "to model."
            )
        return comps

    def _fit_scheffe(self, X: pd.DataFrame, y: np.ndarray, groups) -> None:
        comps = self._scheffe_setup(X)
        self._components_ = comps
        degree = 2 if self.model_type == "scheffe_quadratic" else 1
        self._degree_ = degree
        P = pseudocomponents(X, comps)
        T, names = scheffe_terms(P, comps, degree)
        ok = np.isfinite(T).all(axis=1)
        T, yv = T[ok], y[ok]
        if T.shape[0] < 3:
            raise Refusal("Too few batches with complete mixture components to fit.")
        self.feature_names_ = names
        # small ridge only for numerical stability of the quadratic terms; no intercept,
        # because the components already sum to one
        alphas = np.array([0.0, 1e-6, 1e-4, 1e-2, 1e-1, 1.0])
        splits = self._splits(len(yv), groups.iloc[ok] if groups is not None else None)
        best_a, best = alphas[0], np.inf
        for a in alphas:
            errs = []
            for tr, te in splits:
                tr = tr[tr < len(yv)]
                te = te[te < len(yv)]
                if len(tr) < 2 or len(te) < 1:
                    continue
                A = T[tr].T @ T[tr] + a * np.eye(T.shape[1])
                try:
                    b = np.linalg.solve(A, T[tr].T @ yv[tr])
                except np.linalg.LinAlgError:
                    errs = [np.inf]
                    break
                errs.append(float(np.mean((yv[te] - T[te] @ b) ** 2)))
            score = float(np.mean(errs)) if errs else np.inf
            if score < best - 1e-12:
                best, best_a = score, float(a)
        A = T.T @ T + best_a * np.eye(T.shape[1])
        self.coef_ = np.linalg.solve(A, T.T @ yv)
        self._intercept_ = 0.0
        self.params_ = {"degree": degree, "ridge_alpha": best_a, "intercept": "none (Scheffe canonical form)"}
        self.x_sd_ = np.ones(len(names))
        self.x_mean_ = np.zeros(len(names))
        self.selected_ = list(names)
        self.estimator_ = None

    def component_contrasts(self) -> pd.DataFrame:
        """Pairwise component differences, which is what a mixture coefficient means.

        A Scheffe coefficient is not an effect of adding component i; the components
        sum to a constant, so it is only ever i in place of something else.
        """
        if self.model_type not in MIXTURE_TYPES:
            raise Refusal("Component contrasts are only defined for a mixture model.")
        comps = self._components_
        rows = []
        coef = dict(zip(self.feature_names_, self.coef_))
        for i, a in enumerate(comps):
            for b in comps[i + 1 :]:
                rows.append(
                    {
                        "replacing": b,
                        "with": a,
                        "difference": float(coef.get(a, np.nan) - coef.get(b, np.nan)),
                    }
                )
        return pd.DataFrame(rows).sort_values("difference", key=abs, ascending=False)

    # -- linear models: tune and fit from the regularisation path -------
    def _fit_linear(self, Z: np.ndarray, y: np.ndarray, groups: Optional[pd.Series]) -> None:
        n, p = Z.shape
        splits = self._splits(n, groups)
        if self.model_type == "ridge":
            l1_ratios, alphas, pathfn = [None], RIDGE_ALPHA_GRID, None
        elif self.model_type == "lasso":
            l1_ratios, alphas = [1.0], ALPHA_GRID
        else:
            l1_ratios, alphas = [0.1, 0.5, 0.9], ALPHA_GRID

        best = None
        best_score = np.inf
        for l1 in l1_ratios:
            sse = np.zeros(len(alphas))
            n_used = 0
            for tr, te in splits:
                if len(tr) < 2 or len(te) < 1 or np.std(y[tr]) == 0:
                    continue
                if self.model_type == "ridge":
                    coefs, inter = _ridge_path_coefs(Z[tr], y[tr], alphas)
                else:
                    coefs, inter = _enet_path_coefs(Z[tr], y[tr], l1, alphas)
                pred = Z[te] @ coefs + inter            # (n_test, n_alphas)
                sse += ((y[te][:, None] - pred) ** 2).sum(axis=0)
                n_used += len(te)
            if n_used == 0:
                continue
            mse = sse / n_used
            k = int(np.argmin(mse))
            # ties resolved toward the larger alpha (simpler model): scan from the
            # strong-penalty end, which is the tail of a descending grid
            tol = 1e-12
            candidates = np.where(mse <= mse[k] + tol)[0]
            if len(candidates):
                k = int(candidates.min())   # grid is descending, so min index = max alpha
            if mse[k] < best_score - 1e-12:
                best_score = float(mse[k])
                best = {"alpha": float(alphas[k]), "l1_ratio": l1, "alpha_index": k}
        if best is None:
            best = {"alpha": float(alphas[-1]), "l1_ratio": l1_ratios[0], "alpha_index": len(alphas) - 1}
        self.tuning_score_ = best_score

        # final fit on all training rows, then enforce the complexity cap by moving
        # up the same path rather than by post-hoc truncation
        if self.model_type == "ridge":
            coefs, inter = _ridge_path_coefs(Z, y, alphas)
        else:
            coefs, inter = _enet_path_coefs(Z, y, best["l1_ratio"], alphas)
        k = best["alpha_index"]
        cap = self.cap_["max_features"]
        capped = False
        if self.enforce_cap and self.model_type != "ridge":
            while k > 0 and int(np.sum(np.abs(coefs[:, k]) > 1e-10)) > cap:
                k -= 1                      # descending grid: lower index = larger alpha
                capped = True
        self.params_ = {
            "alpha": float(alphas[k]),
            "l1_ratio": best["l1_ratio"],
            "cap_applied": capped,
        }
        self.estimator_ = _LinearFit(coefs[:, k], inter[k])
        self.coef_ = np.asarray(coefs[:, k], dtype=float)
        self.selected_ = [f for f, c in zip(self.feature_names_, self.coef_) if abs(c) > 1e-10]

    # -- hyperparameter tuning (inner loop only) -----------------------
    def _grid(self, n: int, p: int) -> List[Dict]:
        cap = self.cap_["max_features"]
        if self.model_type in ("elastic_net", "lasso", "ridge"):
            alphas = np.logspace(-3, 1.5, 25)
            if self.model_type == "ridge":
                return [{"alpha": float(a)} for a in np.logspace(-2, 3, 25)]
            l1s = [1.0] if self.model_type == "lasso" else [0.1, 0.5, 0.9]
            return [{"alpha": float(a), "l1_ratio": float(l)} for l in l1s for a in alphas]
        if self.model_type == "pls":
            top = int(max(1, min(cap, p, n - 2)))
            return [{"n_components": k} for k in range(1, top + 1)]
        if self.model_type == "gbm":
            return [
                {"n_estimators": ne, "max_depth": d, "learning_rate": lr}
                for ne in (100, 300)
                for d in (1, min(3, THRESHOLDS["max_depth_small_n"]))
                for lr in (0.05, 0.1)
            ]
        return [
            {"n_estimators": ne, "max_depth": d}
            for ne in (300,)
            for d in (2, THRESHOLDS["max_depth_small_n"])
        ]

    def _make(self, params: Dict):
        if self.model_type == "ridge":
            return Ridge(alpha=params["alpha"], fit_intercept=True, random_state=None)
        if self.model_type in ("elastic_net", "lasso"):
            return ElasticNet(
                alpha=params["alpha"], l1_ratio=params["l1_ratio"], fit_intercept=True,
                max_iter=50000, tol=1e-4, random_state=self.seed, selection="cyclic",
            )
        if self.model_type == "pls":
            return PLSRegression(n_components=params["n_components"], scale=False)
        if self.model_type == "gbm":
            return GradientBoostingRegressor(random_state=self.seed, **params)
        return RandomForestRegressor(random_state=self.seed, n_jobs=1, **params)

    def _splits(self, n: int, groups: Optional[pd.Series]):
        if groups is not None and groups.nunique() >= 2:
            k = int(min(self.inner_cv_splits, groups.nunique()))
            return list(GroupKFold(n_splits=k).split(np.zeros(n), groups=groups.to_numpy()))
        k = int(min(self.inner_cv_splits, n))
        if k < 2:
            return [(np.arange(n), np.arange(n))]
        return list(KFold(n_splits=k, shuffle=True, random_state=self.seed).split(np.zeros(n)))

    def _tune(self, Z: np.ndarray, y: np.ndarray, groups: Optional[pd.Series]) -> Dict:
        n, p = Z.shape
        grid = self._grid(n, p)
        splits = self._splits(n, groups)
        best, best_score = None, np.inf
        for params in grid:
            errs = []
            for tr, te in splits:
                if len(tr) < 2 or len(te) < 1 or np.std(y[tr]) == 0:
                    continue
                try:
                    est = self._make(params)
                    est.fit(Z[tr], y[tr])
                    pred = np.asarray(est.predict(Z[te])).ravel()
                except Exception:  # noqa: BLE001 - a failed hyperparameter is just skipped
                    errs = [np.inf]
                    break
                errs.append(float(np.mean((y[te] - pred) ** 2)))
            score = float(np.mean(errs)) if errs else np.inf
            # ties resolved toward more penalty / fewer components: grid order matters
            if score < best_score - 1e-12:
                best, best_score = params, score
        if best is None:
            best = grid[-1]
        self.tuning_score_ = best_score
        return best

    def _fit_final(self, Z: np.ndarray, y: np.ndarray) -> None:
        params = dict(self.params_)
        est = self._make(params)
        est.fit(Z, y)
        if self.enforce_cap and self.model_type in ("elastic_net", "lasso"):
            cap = self.cap_["max_features"]
            alphas = np.logspace(np.log10(params["alpha"]), 1.8, 40)
            k = 0
            while int(np.sum(np.abs(est.coef_) > 1e-10)) > cap and k < len(alphas):
                params["alpha"] = float(alphas[k])
                est = self._make(params)
                est.fit(Z, y)
                k += 1
            self.params_ = params
        self.estimator_ = est
        self.coef_ = self._extract_coef(est, Z.shape[1])
        self.selected_ = [
            f for f, c in zip(self.feature_names_, self.coef_) if abs(c) > 1e-10
        ]

    def _extract_coef(self, est, p: int) -> np.ndarray:
        if hasattr(est, "coef_"):
            return np.asarray(est.coef_, dtype=float).ravel()[:p]
        if hasattr(est, "feature_importances_"):
            return np.asarray(est.feature_importances_, dtype=float).ravel()[:p]
        return np.zeros(p)

    # -- predict --------------------------------------------------------
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        if not self.fitted_:
            raise RuntimeError("model is not fitted")
        if self.model_type in MIXTURE_TYPES:
            P = pseudocomponents(X, self._components_)
            T, _ = scheffe_terms(P, self._components_, self._degree_)
            T = np.where(np.isfinite(T), T, 0.0)
            return T @ self.coef_ + self._intercept_
        if self.model_type == "group_lasso":
            Z = self._member_matrix(X, train=False)
            return Z @ self._b_members_ + self._intercept_
        F = self._build_features(X, train=False)
        F = F.reindex(columns=self.feature_names_, fill_value=0.0)
        Z = (F.to_numpy(dtype=float) - self.x_mean_) / self.x_sd_
        return np.asarray(self.estimator_.predict(Z)).ravel()

    def per_unit_effect(self, feature: str, member: str) -> float:
        """Response units per one raw unit of `member`, through the feature it sits in.

        Chain: raw -> z-score -> cluster mean -> standardised feature -> coefficient.
        For a cluster this assumes the members move together, which is the only way
        they ever move in this data; that is precisely why they are inseparable.
        """
        if not self.fitted_ or feature not in self.feature_names_:
            return float("nan")
        if self.model_type == "group_lasso":
            # no averaging assumption here: the member has its own coefficient
            b = self._member_coefs_.get(feature, {}).get(member, float("nan"))
            pos = self._raw_cols_.index(member) if member in self._raw_cols_ else None
            if pos is None or not np.isfinite(self._sd_[pos]) or self._sd_[pos] == 0:
                return float("nan")
            return float(b / self._sd_[pos])
        if self.model_type in MIXTURE_TYPES:
            # a mixture coefficient is not a per-unit effect; use component_contrasts
            return float("nan")
        j = self.feature_names_.index(feature)
        if member not in self._raw_cols_:
            return float("nan")
        pos = self._raw_cols_.index(member)
        sd_m = self._sd_[pos]
        if not np.isfinite(sd_m) or sd_m == 0:
            return float("nan")
        k = len(self.plan.groups.get(feature, [feature]))
        return float(self.coef_[j] / self.x_sd_[j] * self._signs_[pos] / (k * sd_m))

    # -- reporting ------------------------------------------------------
    def coefficients(self) -> pd.DataFrame:
        """Coefficients in standardised feature units and, for linear models, in the
        response's own units per SD of the feature."""
        rows = []
        for f, c in zip(self.feature_names_, self.coef_):
            members = self.plan.groups.get(f, [f])
            rows.append(
                {
                    "feature": f,
                    "is_cluster": len(members) > 1,
                    "members": members,
                    "coefficient_std": float(c),
                    "abs_coefficient": abs(float(c)),
                }
            )
        df = pd.DataFrame(rows).sort_values("abs_coefficient", ascending=False)
        return df.reset_index(drop=True)


# ---------------------------------------------------------------------
# stability selection and bootstrap intervals
# ---------------------------------------------------------------------
def bootstrap_indices(
    groups: pd.Series, n_boot: int, seed: int, scheme: str = "batch"
) -> List[np.ndarray]:
    """Deterministic resample indices.

    scheme='batch'        : resample batches with replacement (standard).
    scheme='formulation'  : resample whole formulations (honest for factors that
                            only change between formulations, and much wider).
    """
    rng = np.random.default_rng(seed)
    n = len(groups)
    idx_all = np.arange(n)
    out = []
    if scheme == "formulation":
        levels = pd.unique(groups)
        by_level = {lv: idx_all[(groups == lv).to_numpy()] for lv in levels}
        for _ in range(n_boot):
            for _attempt in range(20):
                pick = rng.choice(levels, size=len(levels), replace=True)
                if len(set(pick)) >= 2:
                    break
            out.append(np.concatenate([by_level[lv] for lv in pick]))
    else:
        for _ in range(n_boot):
            for _attempt in range(20):
                pick = rng.choice(idx_all, size=n, replace=True)
                if len(np.unique(pick)) >= 3:
                    break
            out.append(pick)
    return out


@dataclass
class StabilityResult:
    frequency: pd.Series
    coef_ci: pd.DataFrame
    n_boot: int
    n_effective: int
    scheme: str

    def table(self, threshold: float = THRESHOLDS["stability_unstable"]) -> pd.DataFrame:
        t = self.coef_ci.copy()
        t["selection_frequency"] = self.frequency.reindex(t.index)
        t["stability"] = np.where(
            t["selection_frequency"] >= threshold, "stable", "UNSTABLE"
        )
        return t.sort_values("selection_frequency", ascending=False)


def stability_selection(
    X: pd.DataFrame,
    y: pd.Series,
    groups: pd.Series,
    plan: ClusterPlan,
    model_type: str = "elastic_net",
    n_boot: int = 500,
    seed: int = RANDOM_SEED,
    scheme: str = "batch",
    inner_cv_splits: int = 5,
) -> StabilityResult:
    """Refit everything -- clustering stats, tuning, selection -- in each resample."""
    counts: Dict[str, float] = {}
    coefs: Dict[str, List[float]] = {}
    members_seen: Dict[str, List[str]] = {}
    ok = 0
    for idx in bootstrap_indices(groups, n_boot, seed, scheme):
        Xb, yb, gb = X.iloc[idx], y.iloc[idx], groups.iloc[idx]
        if yb.notna().sum() < 4 or yb.dropna().nunique() < 2:
            continue
        try:
            m = AttributionModel(model_type, plan, seed=seed, inner_cv_splits=inner_cv_splits).fit(Xb, yb, gb)
        except Exception:  # noqa: BLE001
            continue
        ok += 1
        selected = set(m.selected_)
        for f, c in zip(m.feature_names_, m.coef_):
            coefs.setdefault(f, []).append(float(c))
            counts.setdefault(f, 0.0)
            members_seen.setdefault(f, list(m.feature_members_.get(f, [f])))
            if f in selected:
                counts[f] += 1.0
    if ok == 0:
        raise Refusal(
            "No bootstrap resample could be fitted: the response has too little variation or "
            "too few batches. Stability selection cannot be reported."
        )
    feature_names = list(counts)
    freq = pd.Series(counts, dtype=float) / ok
    rows = []
    for f in feature_names:
        v = np.array(coefs[f], dtype=float)
        # features absent from a resample contribute no coefficient; treat as 0
        if len(v) < ok:
            v = np.concatenate([v, np.zeros(ok - len(v))])
        rows.append(
            {
                "feature": f,
                "coef_median": float(np.median(v)) if len(v) else np.nan,
                "coef_lo_2.5": float(np.percentile(v, 2.5)) if len(v) else np.nan,
                "coef_hi_97.5": float(np.percentile(v, 97.5)) if len(v) else np.nan,
                "is_cluster": len(members_seen.get(f, [f])) > 1,
                "members": members_seen.get(f, [f]),
            }
        )
    ci = pd.DataFrame(rows).set_index("feature")
    return StabilityResult(frequency=freq, coef_ci=ci, n_boot=n_boot, n_effective=ok, scheme=scheme)


# ---------------------------------------------------------------------
# the ladder
# ---------------------------------------------------------------------
def ladder_warnings(n: int, model_type: str) -> List[str]:
    w = []
    if model_type == "group_lasso":
        w.append(
            "Group lasso selects or drops an inseparable cluster as a whole. That is the "
            "honest behaviour, and it means a selected cluster still names every member: "
            "the model has not chosen between them and neither should you."
        )
    if model_type in MIXTURE_TYPES:
        w.append(
            "A Scheffe model is fitted on the mixture components only, with no intercept. "
            "Its coefficients are not effects of adding a component: the components sum to "
            "a constant, so every coefficient is that component in place of the others. Read "
            "component_contrasts, not the raw coefficients. Process parameters are not in "
            "this model."
        )
    if model_type in ("gbm", "rf"):
        if n < THRESHOLDS["min_n_for_nonlinear"]:
            w.append(
                f"{model_type} on {n} batches: the model has far more capacity than the data. "
                f"Depth is capped at {THRESHOLDS['max_depth_small_n']}, but expect the "
                "permutation test to be the only thing standing between you and a fiction."
            )
    return w
