"""Prompt 7, tab 8 -- the method record.

Every model, every setting, every rule, with references. Written so that the same
input and the same record reproduce the same output, which is what draft EU GMP
Annex 22 asks of a static, deterministic model.
"""
from __future__ import annotations

import json
from typing import Dict, List

from config import THRESHOLDS, library_versions

REFERENCES: List[Dict[str, str]] = [
    {
        "topic": "False discovery rate on the univariate screen",
        "reference": "Benjamini Y, Hochberg Y. Controlling the false discovery rate. "
        "J R Stat Soc B 1995;57:289-300.",
    },
    {
        "topic": "Elastic net, and why correlated predictors are kept together",
        "reference": "Zou H, Hastie T. Regularization and variable selection via the "
        "elastic net. J R Stat Soc B 2005;67:301-320.",
    },
    {
        "topic": "Group lasso: correlated groups enter or leave together",
        "reference": "Yuan M, Lin Y. Model selection and estimation in regression with "
        "grouped variables. J R Stat Soc B 2006;68:49-67.",
    },
    {
        "topic": "Canonical polynomials for mixture experiments",
        "reference": "Scheffe H. Experiments with mixtures. J R Stat Soc B 1958;20:344-360.",
    },
    {
        "topic": "Stability selection",
        "reference": "Meinshausen N, Buhlmann P. Stability selection. J R Stat Soc B "
        "2010;72:417-473.",
    },
    {
        "topic": "Permutation testing of cross-validated performance",
        "reference": "Ojala M, Garriga GC. Permutation tests for studying classifier "
        "performance. J Mach Learn Res 2010;11:1833-1863.",
    },
    {
        "topic": "Selection bias from selecting once and cross-validating afterwards",
        "reference": "Ambroise C, McLachlan GJ. Selection bias in gene extraction on the "
        "basis of microarray gene-expression data. PNAS 2002;99:6562-6566.",
    },
    {
        "topic": "Nested cross-validation for honest performance estimates",
        "reference": "Varma S, Simon R. Bias in error estimation when using "
        "cross-validation for model selection. BMC Bioinformatics 2006;7:91.",
    },
    {
        "topic": "Y-scrambling in QSAR-style small-n modelling",
        "reference": "Rucker C, Rucker G, Meringer M. y-Randomization and its variants in "
        "QSPR/QSAR. J Chem Inf Model 2007;47:2345-2357.",
    },
    {
        "topic": "PLS for correlated pharmaceutical predictors",
        "reference": "Wold S, Sjostrom M, Eriksson L. PLS-regression: a basic tool of "
        "chemometrics. Chemom Intell Lab Syst 2001;58:109-130.",
    },
    {
        "topic": "Mixture constraints in formulation data",
        "reference": "Cornell JA. Experiments with Mixtures. 3rd ed. Wiley, 2002.",
    },
    {
        "topic": "Dissolution profile comparison and f2",
        "reference": "FDA Guidance for Industry: Dissolution Testing of Immediate Release "
        "Solid Oral Dosage Forms, 1997; EMA CHMP/EWP/QWP/1401/98 Rev.1.",
    },
    {
        "topic": "Model-independent profile metrics (MDT, dissolution efficiency)",
        "reference": "Costa P, Lobo JMS. Modeling and comparison of dissolution profiles. "
        "Eur J Pharm Sci 2001;13:123-133.",
    },
    {
        "topic": "Content uniformity acceptance value",
        "reference": "USP <905> Uniformity of Dosage Units.",
    },
    {
        "topic": "Status of this application under GMP",
        "reference": "EU GMP Annex 22 (draft, 2025): models in critical applications must "
        "be static and deterministic with documented human oversight and change control. "
        "This application is a development and investigation aid, not a validated GMP output.",
    },
]

DECISION_RULES = {
    "columns excluded": "constant across all batches, entirely missing, or non-numeric in a "
    "numeric family; each with its reason recorded, never dropped silently",
    "p > n/3": f"flagged as unstable variable selection (threshold {THRESHOLDS['p_over_n_flag']:.3f})",
    "inseparable cluster": f"|r| >= {THRESHOLDS['collinear_cluster_r']} on the predictor "
    "correlation graph, reported and modelled as one feature",
    "aliased with formulation": f"R-squared > {THRESHOLDS['formulation_alias_r2']} regressing "
    "the predictor on formulation identity",
    "aliased with time": f"R-squared > {THRESHOLDS['time_alias_r2']} regressing the predictor "
    "on manufacturing date",
    "mixture constraint": f"component percentages summing to a constant within "
    f"{THRESHOLDS['mixture_sum_tolerance']}; one component removed as the dependent remainder",
    "complexity cap": "max_features = clip(floor(n_train/5), 1, 10), enforced by moving up the "
    "regularisation path inside every training fold",
    "cross-validation": "grouped only; leave-one-formulation-out by default. No random row "
    "split is reachable for reporting",
    "hyperparameter tuning": "inner loop only, refitted inside every outer fold, every "
    "bootstrap resample and every permutation",
    "variable selection": "refitted inside every resample; never selected once on all data "
    "and cross-validated afterwards",
    "permutation test": "response shuffled, whole pipeline refitted; empirical "
    f"p = (1 + #{{null >= observed}}) / (n + 1); gate at p <= {THRESHOLDS['permutation_alpha']}",
    "stability threshold": f"selection frequency below {THRESHOLDS['stability_unstable']:.0%} "
    "labelled UNSTABLE and not treated as a finding",
    "non-linear models": f"offered below n = {THRESHOLDS['min_n_for_nonlinear']} only with a "
    f"warning; depth capped at {THRESHOLDS['max_depth_small_n']}",
    "findings gate": "no importance ranking, partial dependence or contribution plot is "
    "produced when the permutation test failed",
    "group lasso penalty": "sqrt(rank) weights on orthonormalised blocks, blocks taken from "
    "the correlation clusters; a cluster is selected or dropped as a whole and the cap "
    "counts groups, not members",
    "dense model selection": "ridge, PLS, Scheffe, GBM and RF have no zero coefficients, so "
    "'selected' means inside the complexity cap by absolute effect; reported as such rather "
    "than as a sparsity frequency",
    "Scheffe mixture model": "fitted on the mixture components only, rescaled to "
    "pseudocomponents summing to one, with NO intercept; coefficients are contrasts between "
    "components, never per-unit effects, and process and material factors are absent from "
    "the model and are not tested by it",
}


def build_method_record(result) -> Dict:
    """Assemble the full record from an AnalysisResult."""
    rec = dict(result.method_record)
    rec["libraries"] = library_versions()
    rec["decision_rules"] = DECISION_RULES
    rec["thresholds"] = THRESHOLDS
    rec["references"] = REFERENCES
    rec["design_assessment"] = result.design.statements
    rec["response_assessment"] = result.response.statements
    rec["refusals"] = result.refusals
    rec["warnings"] = result.warnings
    if result.validation is not None:
        v = result.validation
        rec["validation"] = {
            "scheme": v.cv["scheme"],
            "q2": v.cv["q2"],
            "rmse": v.cv["rmse"],
            "n_used": v.cv["n_used"],
            "folds": v.cv["per_fold"].to_dict("records") if len(v.cv["per_fold"]) else [],
            "permutation_p": v.permutation.get("p_value"),
            "permutation_n": v.permutation.get("n_permutations"),
            "permutation_null_p95": v.permutation.get("null_p95"),
            "random_split_q2_for_comparison_only": v.random_split_q2,
            "stability_n_effective": v.stability.n_effective,
            "verdict": v.verdict,
        }
    if result.report is not None:
        rec["headline"] = result.report.headline
        rec["findings"] = [
            {
                "feature": f.feature,
                "members": f.members,
                "coefficient_per_sd": f.coefficient_std,
                "ci": [f.ci_low, f.ci_high],
                "selection_frequency": f.selection_frequency,
                "confounds": f.confounds,
                "varies_within_formulation": f.varies_within_formulation,
            }
            for f in result.report.findings
        ]
        if result.report.experiment:
            rec["proposed_experiment"] = result.report.experiment.text()
    return rec


def to_json(result) -> str:
    def default(o):
        try:
            import numpy as np

            if isinstance(o, (np.integer,)):
                return int(o)
            if isinstance(o, (np.floating,)):
                return float(o)
            if isinstance(o, np.ndarray):
                return o.tolist()
        except Exception:  # noqa: BLE001
            pass
        return str(o)

    return json.dumps(build_method_record(result), indent=2, default=default)


def to_markdown(result) -> str:
    rec = build_method_record(result)
    lines = ["# Method record", ""]
    lines.append(
        "This application is a development and investigation aid. Nothing it produces is a "
        "validated GMP output."
    )
    lines += ["", "## Run", ""]
    for k in ("model_type", "cv_scheme", "n_batches", "n_formulations",
              "n_candidate_predictors", "n_model_features"):
        if k in rec:
            lines.append(f"- **{k}**: {rec[k]}")
    lines.append(f"- **settings**: `{rec.get('settings')}`")
    lines.append(f"- **libraries**: `{rec.get('libraries')}`")

    if rec.get("clusters"):
        lines += ["", "## Inseparable clusters modelled as single features", ""]
        for name, members in rec["clusters"].items():
            lines.append(f"- `{name}`: {', '.join(members)}")

    for title, key in (
        ("Columns excluded at load", "excluded_columns"),
        ("Zero-information columns", "zero_information_columns"),
        ("Removed to break a mixture or linear dependency", "mixture_dropped"),
        ("Predictors excluded for missingness", "predictors_excluded_for_missingness"),
        ("Imputation performed", "imputation"),
    ):
        val = rec.get(key)
        if val:
            lines += ["", f"## {title}", ""]
            if isinstance(val, dict):
                lines += [f"- `{k}`: {v}" for k, v in val.items()]
            else:
                lines += [f"- {v}" for v in val]

    lines += ["", "## Design assessment", ""]
    lines += [f"- {s}" for s in rec.get("design_assessment", [])]
    lines += ["", "## Response assessment", ""]
    lines += [f"- {s}" for s in rec.get("response_assessment", [])] or ["- none"]

    if rec.get("refusals"):
        lines += ["", "## Refusals", ""] + [f"- {s}" for s in rec["refusals"]]
    if rec.get("warnings"):
        lines += ["", "## Warnings", ""] + [f"- {s}" for s in rec["warnings"]]
    if rec.get("validation"):
        lines += ["", "## Validation", ""]
        lines += [f"- **{k}**: {v}" for k, v in rec["validation"].items() if k != "folds"]
    if rec.get("headline"):
        lines += ["", "## Conclusion", "", rec["headline"]]
    if rec.get("proposed_experiment"):
        lines += ["", "## Proposed confirmatory experiment", "", "```",
                  rec["proposed_experiment"], "```"]

    lines += ["", "## Decision rules applied", ""]
    lines += [f"- **{k}**: {v}" for k, v in DECISION_RULES.items()]
    lines += ["", "## References", ""]
    lines += [f"- *{r['topic']}* — {r['reference']}" for r in REFERENCES]
    return "\n".join(lines)
