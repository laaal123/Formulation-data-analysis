"""Prompt 6 -- interpretation.

Findings are hypotheses with their confounders attached, or they are not shown.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from config import THRESHOLDS
from diagnostics import DesignAssessment
from modelling import AttributionModel, ClusterPlan
from validation import ValidationResult


# ---------------------------------------------------------------------
# the causal-language linter (prompt 6 forbids these outputs)
# ---------------------------------------------------------------------
CAUSAL_PATTERNS = [
    r"\bcaus(e|es|ed|ing)\b",
    r"\bdue to\b",
    r"\bbecause of\b",
    r"\bdriven by\b",
    r"\bdriver of\b",
    r"\bresponsible for\b",
    r"\bleads? to\b",
    r"\bresults? in\b",
    r"\bimpact(s|ed)? the\b",
    r"\beffect of .* on\b",
]
_ALLOWED_CONTEXT = (
    "cannot establish caus",
    "not causation",
    "would establish caus",
    "to establish which",
    "establish causation",
)


def causal_language_report(text: str) -> List[str]:
    """Return the offending fragments. Used by the app and by the test suite."""
    hits = []
    low = text.lower()
    for pat in CAUSAL_PATTERNS:
        for m in re.finditer(pat, low):
            start = max(0, m.start() - 60)
            frag = low[start : m.end() + 40]
            if any(a in frag for a in _ALLOWED_CONTEXT):
                continue
            hits.append(low[max(0, m.start() - 30) : m.end() + 30].strip())
    return hits


# ---------------------------------------------------------------------
# findings
# ---------------------------------------------------------------------
@dataclass
class Finding:
    feature: str
    members: List[str]
    is_cluster: bool
    coefficient_std: float                  # response units per 1 SD of the feature
    ci_low: float
    ci_high: float
    selection_frequency: float
    confounds: List[str]
    varies_within_formulation: bool
    formulation_alias_r2: float
    per_unit: Dict[str, float] = field(default_factory=dict)   # member -> response units per raw unit
    member_sd: Dict[str, float] = field(default_factory=dict)
    effect_kind: str = "per_sd"            # per_sd | mixture_component
    selection_label: str = "selected"      # dense models select by rank, not by sparsity
    contrast_against: str = ""             # for mixture findings

    @property
    def stable(self) -> bool:
        return bool(self.selection_frequency >= THRESHOLDS["stability_unstable"])

    def sentence(self, response_name: str, response_units: str = "") -> str:
        unit = f" {response_units}" if response_units else ""
        head = self.members[0]
        per = self.per_unit.get(head, np.nan)
        sd = self.member_sd.get(head, np.nan)
        if self.effect_kind == "mixture_component":
            # a mixture coefficient is never a per-unit or per-SD effect: the components
            # sum to a constant, so it only ever means this component in place of another
            against = self.contrast_against or "the remaining components"
            bits = [
                f"replacing {against} with {head} at constant total gives a Scheffe "
                f"coefficient of {self.coefficient_std:.3g}{unit} on the 0 to 1 component "
                f"scale, which is a contrast and not an effect of adding {head}",
                f"bootstrap 95% interval {self.ci_low:.3g} to {self.ci_high:.3g}",
                f"{self.selection_label} in {self.selection_frequency:.0%} of resamples",
            ]
            if not self.stable:
                bits.append("UNSTABLE - below the 60% threshold, not a finding")
            bits.append(
                ("cannot be separated from: " + ", ".join(self.confounds))
                if self.confounds
                else "no factor was found to be inseparable from it"
            )
            bits.append(
                "this model contains the mixture components only; process and material "
                "factors were not in it and are not ruled out by it"
            )
            return ". ".join(bits) + "."
        if np.isfinite(per) and np.isfinite(sd) and sd > 0:
            step = _nice_step(sd)
            mag = per * step
            direction = "an increase" if mag > 0 else "a decrease"
            magnitude = (
                f"+{step:g} in {head} is associated with {direction} of "
                f"{abs(mag):.3g}{unit} in {response_name}"
            )
        else:
            magnitude = (
                f"1 SD of {self.feature} is associated with a change of "
                f"{self.coefficient_std:.3g}{unit} in {response_name}"
            )
        bits = [magnitude]
        bits.append(
            f"bootstrap 95% interval {self.ci_low:.3g} to {self.ci_high:.3g} "
            f"(response units per SD)"
        )
        bits.append(f"{self.selection_label} in {self.selection_frequency:.0%} of resamples")
        if not self.stable:
            bits.append("UNSTABLE - below the 60% selection threshold, not a finding")
        if self.confounds:
            bits.append("cannot be separated from: " + ", ".join(self.confounds))
        else:
            bits.append("no factor was found to be inseparable from it")
        bits.append(
            "varies within formulations, so it is estimable from batch-to-batch variation"
            if self.varies_within_formulation
            else "changes only between formulations, so it carries the whole formulation "
            "change with it"
        )
        return ". ".join(bits) + "."


def _nice_step(sd: float) -> float:
    """A round increment of about one SD, for readable sentences."""
    if not np.isfinite(sd) or sd <= 0:
        return 1.0
    mag = 10 ** np.floor(np.log10(sd))
    for mult in (1, 2, 5, 10):
        if mult * mag >= sd:
            return float(mult * mag)
    return float(10 * mag)


def build_findings(
    model: AttributionModel,
    validation: ValidationResult,
    design: DesignAssessment,
    X: pd.DataFrame,
    top_n: int = 10,
) -> List[Finding]:
    """Only ever called after the permutation gate; see build_report."""
    stab = validation.stability
    table = stab.table()
    coefs = dict(zip(model.feature_names_, model.coef_))
    findings: List[Finding] = []
    for feature, row in table.iterrows():
        coef = float(coefs.get(feature, np.nan))
        members = list(getattr(model, 'feature_members_', {}).get(feature, model.plan.groups.get(feature, [feature])))
        if abs(coef) <= 1e-10 and float(row["selection_frequency"]) < THRESHOLDS["stability_unstable"]:
            continue
        member_sd, per_unit = {}, {}
        for m in members:
            if m in X.columns:
                sd = float(pd.to_numeric(X[m], errors="coerce").std(ddof=0))
                member_sd[m] = sd
                per_unit[m] = model.per_unit_effect(feature, m)
        alias = float(design.formulation_alias_r2.get(members[0], np.nan))
        confounds = sorted({c for m in members for c in design.confounds_for(m)} - set(members))
        from modelling import MIXTURE_TYPES, SPARSE_TYPES

        is_mixture = model.model_type in MIXTURE_TYPES
        if model.model_type in SPARSE_TYPES:
            label = "selected"
        else:
            cap = model.cap_["max_features"]
            label = f"inside the top {cap} by absolute effect"
        against = ""
        if is_mixture:
            others = [c for c in getattr(model, "_components_", []) if c not in members]
            against = others[0] if len(others) == 1 else "the other components"
        findings.append(
            Finding(
                feature=feature,
                members=members,
                is_cluster=len(members) > 1,
                coefficient_std=coef,
                ci_low=float(row["coef_lo_2.5"]),
                ci_high=float(row["coef_hi_97.5"]),
                selection_frequency=float(row["selection_frequency"]),
                confounds=confounds,
                varies_within_formulation=bool(
                    np.isfinite(alias) and alias <= THRESHOLDS["formulation_alias_r2"]
                ),
                formulation_alias_r2=alias,
                per_unit=per_unit,
                member_sd=member_sd,
                effect_kind="mixture_component" if is_mixture else "per_sd",
                selection_label=label,
                contrast_against=against,
            )
        )
    findings.sort(key=lambda f: (-f.selection_frequency, -abs(f.coefficient_std)))
    return findings[:top_n]


# ---------------------------------------------------------------------
# gated plots
# ---------------------------------------------------------------------
def partial_dependence(
    model: AttributionModel, X: pd.DataFrame, member: str, validation: ValidationResult,
    grid_points: int = 20,
) -> pd.DataFrame:
    """Partial dependence of the response on one raw column.

    Refuses to compute anything when the permutation test failed: a dependence plot
    from a model that did not beat its null is a picture of noise.
    """
    if not validation.passed:
        raise PermissionError(
            "Partial dependence is not available: the model did not beat its permutation "
            "null, so no dependence it shows is distinguishable from chance."
        )
    if member not in X.columns:
        raise KeyError(member)
    x = pd.to_numeric(X[member], errors="coerce")
    grid = np.linspace(np.nanpercentile(x, 2), np.nanpercentile(x, 98), grid_points)
    out = []
    for v in grid:
        Xv = X.copy()
        Xv[member] = v
        out.append({"value": float(v), "prediction": float(np.mean(model.predict(Xv)))})
    return pd.DataFrame(out)


def linear_contributions(
    model: AttributionModel, X: pd.DataFrame, validation: ValidationResult
) -> pd.DataFrame:
    """Exact per-batch contributions for a linear model.

    For a linear model these ARE the Shapley values, computed in closed form; no
    approximation and no extra dependency. Gated on the permutation test.
    """
    if not validation.passed:
        raise PermissionError(
            "Contribution values are not available: the permutation test did not pass."
        )
    F = model._build_features(X, train=False).reindex(columns=model.feature_names_, fill_value=0.0)
    Z = (F.to_numpy(dtype=float) - model.x_mean_) / model.x_sd_
    contrib = Z * model.coef_
    return pd.DataFrame(contrib, index=X.index, columns=model.feature_names_)


# ---------------------------------------------------------------------
# the confirmatory experiment
# ---------------------------------------------------------------------
@dataclass
class ProposedExperiment:
    factors: List[Dict]
    hold_constant: List[str]
    design_name: str
    n_runs: int
    n_centre_points: int
    confirmation_criterion: str

    def text(self) -> str:
        lines = [f"Proposed design: {self.design_name}."]
        for f in self.factors:
            lines.append(
                f"  Vary {f['name']} from {f['low']:.3g} to {f['high']:.3g}"
                + (" " + f["units"] if f.get("units") else "")
                + f" (observed range {f['observed_low']:.3g} to {f['observed_high']:.3g})."
            )
        if self.hold_constant:
            lines.append("  Hold constant: " + ", ".join(self.hold_constant) + ".")
        lines.append(
            f"  {self.n_runs} batches including {self.n_centre_points} centre point(s)."
        )
        lines.append("  " + self.confirmation_criterion)
        return "\n".join(lines)


SETTABLE_FAMILIES = ("FORMULATION", "PROCESS", "MATERIAL")


def _settable(name: str, families: Optional[Dict[str, str]]) -> bool:
    """An IPQC result is measured, not set. You cannot run a factorial in hardness."""
    if not families:
        return True
    return families.get(name) in SETTABLE_FAMILIES


def propose_experiment(
    finding: Finding,
    X: pd.DataFrame,
    design: DesignAssessment,
    response_name: str,
    families: Optional[Dict[str, str]] = None,
    max_factors: int = 3,
) -> ProposedExperiment:
    """The smallest design that separates the top finding from its confounders.

    Only factors that can actually be set are proposed, and two factors that are the
    same quantity in different units (r near 1) are never both listed, because no
    design can vary them independently.
    """
    pool = [finding.members[0]] + [c for c in finding.confounds if c in X.columns]
    pool = [c for c in dict.fromkeys(pool) if c in X.columns]
    measured_only = [c for c in pool if not _settable(c, families)]
    settable = [c for c in pool if _settable(c, families)]
    if not settable:
        settable = pool[:1]

    mixture_partners = {}
    for g in design.mixture_groups:
        for c in g["columns"]:
            mixture_partners.setdefault(c, set()).update(set(g["columns"]) - {c})

    candidates: List[str] = []
    remainder_of_mixture: List[str] = []
    for c in settable:
        dup = False
        if any(c in mixture_partners.get(chosen, set()) for chosen in candidates):
            # in a mixture the components sum to a constant, so this one moves as the
            # remainder when the chosen factor is set; it is not a separate knob
            remainder_of_mixture.append(c)
            continue
        a = pd.to_numeric(X[c], errors="coerce")
        for chosen in candidates:
            b = pd.to_numeric(X[chosen], errors="coerce")
            m = a.notna() & b.notna()
            if m.sum() > 2 and a[m].std() > 0 and b[m].std() > 0:
                if abs(float(np.corrcoef(a[m], b[m])[0, 1])) > 0.995:
                    dup = True          # the same quantity in different units
                    break
        if not dup:
            candidates.append(c)
        if len(candidates) >= max_factors:
            break
    factors = []
    for name in candidates:
        x = pd.to_numeric(X[name], errors="coerce").dropna()
        lo, hi = float(x.min()), float(x.max())
        span = hi - lo
        if span <= 0:
            lo, hi, span = lo * 0.9, lo * 1.1, abs(lo * 0.2)
        factors.append(
            {
                "name": name,
                "low": lo,
                "high": hi,
                "observed_low": lo,
                "observed_high": hi,
                "centre": (lo + hi) / 2,
            }
        )
    k = len(factors)
    if k == 0:
        raise ValueError("No usable factor to design around")
    if k <= 3:
        n_centre = 2
        design_name = f"{k}-factor two-level full factorial (2^{k}) with centre points"
        n_runs = 2**k + n_centre
    else:
        n_centre = 3
        design_name = f"definitive screening design in {k} factors"
        n_runs = 2 * k + 1 + n_centre
    chosen = [f["name"] for f in factors]
    hold = [
        c for c in design.within_formulation_estimable
        if c not in chosen and _settable(c, families)
    ][:6]
    hold += [c + " (measured, expected to follow)" for c in measured_only[:3]]
    hold += [
        c + " (moves as the mixture remainder)" for c in remainder_of_mixture[:3]
    ]
    criterion = (
        f"Confirmation: {response_name} must move with {factors[0]['name']} at both levels "
        f"of the other factor(s), with the {factors[0]['name']} main effect exceeding the "
        "centre-point replicate standard deviation by a factor of three. If the response "
        f"instead tracks {factors[1]['name'] if k > 1 else 'the other factor'}, the "
        "attribution belongs there."
    )
    return ProposedExperiment(
        factors=factors,
        hold_constant=hold,
        design_name=design_name,
        n_runs=n_runs,
        n_centre_points=n_centre,
        confirmation_criterion=criterion,
    )


# ---------------------------------------------------------------------
# the report
# ---------------------------------------------------------------------
@dataclass
class Report:
    verdict: str
    headline: str
    findings: List[Finding]
    experiment: Optional[ProposedExperiment]
    banner: str = (
        "Association from observational batch data. Not causation. "
        "See the confound list for each factor."
    )
    statements: List[str] = field(default_factory=list)

    def text(self) -> str:
        parts = [self.banner, "", self.headline, ""]
        for f in self.findings:
            parts.append("- " + f.sentence(self.statements[0] if self.statements else "the response"))
        if self.experiment:
            parts += ["", self.experiment.text()]
        return "\n".join(parts)


def build_report(
    model: Optional[AttributionModel],
    validation: ValidationResult,
    design: DesignAssessment,
    X: pd.DataFrame,
    response_name: str,
    response_units: str = "",
    families: Optional[Dict[str, str]] = None,
) -> Report:
    """The only supported way to produce findings. Refuses when validation failed."""
    n = design.n_batches
    p = design.n_predictors
    perm = validation.permutation
    q2 = validation.cv["q2"]

    if not validation.passed:
        headline = (
            f"No attribution supported. With {n} batches and {p} candidate predictors, the "
            f"best model achieved a cross-validated Q-squared of {q2:.2f} against a "
            f"permutation null whose 95th percentile is {perm.get('null_p95', float('nan')):.2f} "
            f"(p = {perm.get('p_value', float('nan')):.2f}). The apparent relationships in the "
            f"raw data do not survive held-out validation. "
            f"{len(design.aliased_with_formulation)} predictor(s) are aliased with formulation "
            "identity and cannot be assessed independently in any case."
        )
        exp = None
        # a confirmatory design is still the useful answer, built from the strongest
        # inseparable group rather than from a model result
        if design.inseparable_groups:
            grp = design.inseparable_groups[0]
            pseudo = Finding(
                feature=grp[0], members=[grp[0]], is_cluster=False, coefficient_std=np.nan,
                ci_low=np.nan, ci_high=np.nan, selection_frequency=np.nan,
                confounds=[c for c in grp[1:]], varies_within_formulation=False,
                formulation_alias_r2=float(design.formulation_alias_r2.get(grp[0], np.nan)),
            )
            try:
                exp = propose_experiment(pseudo, X, design, response_name, families)
            except Exception:  # noqa: BLE001
                exp = None
        return Report(
            verdict=validation.verdict,
            headline=headline,
            findings=[],
            experiment=exp,
            statements=[response_name],
        )

    findings = build_findings(model, validation, design, X)
    if not findings:
        return Report(
            verdict=validation.verdict,
            headline=(
                "The model beat its permutation null but retained no predictor at any "
                "meaningful stability. There is a signal in the response that this predictor "
                "set does not localise to any factor."
            ),
            findings=[],
            experiment=None,
            statements=[response_name],
        )

    top = findings[0]
    exp = propose_experiment(top, X, design, response_name, families)
    confound_txt = ", ".join(top.confounds) if top.confounds else "no other factor"
    from modelling import MIXTURE_TYPES

    scope = ""
    if model.model_type in MIXTURE_TYPES:
        omitted = [c for c in X.columns if c not in getattr(model, "_components_", [])]
        scope = (
            f" This is a Scheffe mixture model fitted on the {len(getattr(model, '_components_', []))} "
            f"formulation components only. The other {len(omitted)} candidate predictors, "
            "including every process and material factor, were not in the model, so nothing "
            "here weighs the components against them or rules them out."
        )
    headline = (
        f"{response_name} is most strongly associated with {top.feature} "
        f"(CV Q-squared {q2:.2f}, permutation p = {perm['p_value']:.3f}, "
        f"{top.selection_label} in {top.selection_frequency:.0%} of resamples). "
        f"However {top.feature} is confounded with {confound_txt} across all available "
        f"batches. To establish which is responsible, run {exp.design_name}.{scope}"
    )
    return Report(
        verdict=validation.verdict,
        headline=headline,
        findings=findings,
        experiment=exp,
        statements=[response_name],
    )
