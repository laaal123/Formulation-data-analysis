"""End-to-end orchestration in the mandated order.

diagnostics -> response -> model -> validation -> findings. Findings are
unreachable except through validation; there is no other entry point.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from config import RunSettings, library_versions
from diagnostics import DesignAssessment, assess_design, mixture_safe_predictors
from errors import Refusal
from interpret import Report, build_report, causal_language_report
from loader import BatchDataset
from modelling import (
    AttributionModel,
    ClusterPlan,
    ladder_warnings,
    univariate_screen,
)
from response import ResponseResult, build_response
from validation import ValidationResult, run_validation


@dataclass
class AnalysisResult:
    design: DesignAssessment
    response: ResponseResult
    plan: ClusterPlan
    model_input: pd.DataFrame
    screen: pd.DataFrame
    model: Optional[AttributionModel]
    validation: Optional[ValidationResult]
    report: Optional[Report]
    method_record: Dict
    refusals: List[str] = field(default_factory=list)
    warnings: List[str] = field(default_factory=list)

    @property
    def refused(self) -> bool:
        return self.validation is None

    def text(self) -> str:
        parts = ["=== DESIGN ASSESSMENT ===", self.design.text(),
                 "", "=== RESPONSE ===", self.response.text()]
        if self.refusals:
            parts += ["", "=== REFUSED ===", *self.refusals]
        if self.validation is not None:
            parts += ["", "=== VALIDATION ===", self.validation.text()]
        if self.report is not None:
            parts += ["", "=== FINDINGS ===", self.report.text()]
        return "\n".join(parts)


def run_attribution(
    ds: BatchDataset,
    response_column: Optional[str] = None,
    *,
    profile_metric: Optional[str] = None,
    medium: Optional[str] = None,
    reference_batch: Optional[str] = None,
    method_rsd_pct: Optional[float] = None,
    response_units: str = "",
    model_type: str = "elastic_net",
    scheme: str = "leave_one_formulation_out",
    settings: Optional[RunSettings] = None,
    predictors: Optional[List[str]] = None,
    allow_missing_predictors: bool = False,
    n_jobs: int = 1,
) -> AnalysisResult:
    st = settings or RunSettings()
    refusals: List[str] = []
    warnings: List[str] = []

    # 1. design diagnostics first, always
    cols = list(predictors) if predictors is not None else ds.predictor_columns()
    if response_column in cols:
        cols.remove(response_column)          # never predict a thing with itself
    try:
        design = assess_design(ds, cols)
    except Refusal as exc:
        raise

    # 2. response
    response = build_response(
        ds,
        response_column,
        metric=profile_metric,
        medium=medium,
        reference_batch=reference_batch,
        method_rsd_pct=method_rsd_pct,
        units=response_units,
    )

    live = [c for c in cols if c not in design.zero_information]

    # Predictors with missing values are dropped unless the user has explicitly
    # imputed them. Otherwise the model's internal fallback (a feature set to the
    # training mean) would be a silent imputation, which is exactly what prompt 1
    # forbids.
    incomplete = [c for c in live if ds.wide[c].isna().any()]
    if incomplete and not allow_missing_predictors:
        live = [c for c in live if c not in incomplete]
        warnings.append(
            f"{len(incomplete)} predictor(s) have missing values and were excluded rather "
            "than imputed: "
            + ", ".join(incomplete[:8])
            + ("..." if len(incomplete) > 8 else "")
            + ". Choose an imputation method explicitly (it will be recorded), or pass "
            "allow_missing_predictors=True to accept the training-mean fallback."
        )
    kept, mixture_dropped = mixture_safe_predictors(ds, design, live)
    plan = ClusterPlan.from_design(design, live, mixture_dropped)
    X = ds.wide[live]
    y = response.values.reindex(ds.wide.index)
    groups = ds.formulations

    screen = univariate_screen(X.loc[y.notna()], y.dropna()) if response.usable else pd.DataFrame()

    record = {
        "settings": st.to_dict(),
        "predictors_excluded_for_missingness": incomplete if not allow_missing_predictors else [],
        "model_type": model_type,
        "cv_scheme": scheme,
        "libraries": library_versions(),
        "n_batches": ds.n_batches,
        "n_formulations": ds.n_formulations,
        "n_candidate_predictors": len(cols),
        "n_model_features": len(plan.feature_names),
        "clusters": {k: v for k, v in plan.groups.items() if len(v) > 1},
        "mixture_dropped": mixture_dropped,
        "excluded_columns": dict(ds.excluded),
        "zero_information_columns": design.zero_information,
        "imputation": list(ds.imputation_record),
        "response": {"name": response.name, "source": response.source,
                     "kind": response.kind, "diagnostics_keys": sorted(response.diagnostics)},
        "complexity_rule": "max_features = clip(floor(n_train/5), 1, 10), applied per training fold",
    }

    # -- refusal gates -------------------------------------------------
    if not response.usable:
        refusals.append(
            "Refused to model: the response did not pass its own checks. " + response.text()
        )
    if int(y.notna().sum()) < 6:
        refusals.append(
            f"Refused to model: only {int(y.notna().sum())} batches have this response. "
            "Cross-validation of any kind would be meaningless."
        )
    if ds.n_formulations < 2 and scheme.startswith("leave_one_formulation"):
        refusals.append(
            "Refused to validate by leave-one-formulation-out: there is only one "
            "formulation, so no formulation can be held out. Nothing about formulation "
            "differences can be assessed from this dataset."
        )
    if not plan.feature_names:
        refusals.append("Refused to model: no predictor carries any information.")

    warnings += ladder_warnings(int(y.notna().sum()), model_type)
    warnings += design.notes

    if refusals:
        return AnalysisResult(
            design=design, response=response, plan=plan, model_input=X, screen=screen,
            model=None, validation=None, report=None, method_record=record,
            refusals=refusals, warnings=warnings,
        )

    # 3. validation (which fits the model many times over)
    try:
        validation = run_validation(X, y, groups, plan, model_type, scheme, st, n_jobs=n_jobs)
    except Refusal as exc:
        refusals.append(f"Refused during validation: {exc}")
        return AnalysisResult(
            design=design, response=response, plan=plan, model_input=X, screen=screen,
            model=None, validation=None, report=None, method_record=record,
            refusals=refusals, warnings=warnings,
        )
    warnings += validation.warnings

    # 4. one final model on all data, used only for effect direction and plots,
    #    and only reachable if validation passed
    model = AttributionModel(model_type, plan, seed=st.seed, inner_cv_splits=st.inner_cv_splits)
    model.fit(X, y, groups)
    record["final_model_params"] = dict(model.params_)
    record["complexity_cap_final"] = model.cap_
    record["feature_cells_filled_with_training_mean"] = int(getattr(model, "filled_cells_", 0))
    if getattr(model, "filled_cells_", 0):
        warnings.append(
            f"{model.filled_cells_} feature value(s) were missing at fit time and fell back to "
            f"the training mean: {getattr(model, 'filled_features_', [])[:6]}."
        )

    report = build_report(
        model, validation, design, X, response.name, response_units, ds.families
    )

    # 5. self-check: the app must not emit causal language
    offenders = causal_language_report(report.headline + " " + report.text())
    if offenders:
        warnings.append(
            "Internal check: causal phrasing detected in the generated report and should be "
            f"reported as a defect: {offenders[:3]}"
        )

    return AnalysisResult(
        design=design, response=response, plan=plan, model_input=X, screen=screen,
        model=model, validation=validation, report=report, method_record=record,
        refusals=refusals, warnings=warnings,
    )
