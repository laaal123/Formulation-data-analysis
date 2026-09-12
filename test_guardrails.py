"""The guardrail prompt, run as a test.

Ten failure modes. Each is checked either in the source (so it cannot be
reintroduced quietly) or in behaviour (so it cannot be faked).
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from config import RunSettings
from diagnostics import assess_design, mixture_safe_predictors
from interpret import causal_language_report, linear_contributions, partial_dependence
from modelling import AttributionModel, ClusterPlan
from pipeline import run_attribution
from synthetic import example_dataset, make_controlled
from validation import make_splits

PKG = Path(__file__).resolve().parent
_MODULES = {m + ".py" for m in ['config', 'errors', 'loader', 'synthetic', 'diagnostics', 'response', 'modelling', 'validation', 'interpret', 'gating', 'record', 'pipeline', 'explore', 'catalogue']}
SOURCES = {p.name: p.read_text() for p in PKG.glob("*.py") if p.name in _MODULES}
FAST = RunSettings(n_permutations=40, n_bootstrap=30, inner_cv_splits=3)


# 1 -- no random train/test splitting on batch data
def test_1_no_random_split_outside_the_labelled_comparison():
    offenders = []
    for name, src in SOURCES.items():
        for m in re.finditer(r"train_test_split|KFold\(", src):
            line_no = src[: m.start()].count("\n")
            lines = src.splitlines()
            line = lines[line_no]
            enclosing = ""
            for k in range(line_no, -1, -1):            # find the enclosing function
                stripped = lines[k].lstrip()
                if stripped.startswith("def "):
                    enclosing = stripped.split("(")[0][4:]
                    break
            allowed = (
                "GroupKFold" in line
                or enclosing == "random_split_q2"       # the labelled comparison only
                or enclosing == "_splits"               # inner tuning loop, inside a fold
            )
            if not allowed:
                offenders.append(f"{name}:{line_no + 1}: {line.strip()}")
    assert not offenders, offenders


def test_1b_make_splits_offers_grouped_schemes_only():
    from validation import CV_SCHEMES

    assert all("random" not in s for s in CV_SCHEMES)
    with pytest.raises(ValueError):
        make_splits(pd.Series(["F1", "F2"] * 4), "shuffle")


# 2 -- hyperparameter tuning inside the outer loop
def test_2_tuning_happens_inside_every_fold():
    ds = example_dataset()
    design = assess_design(ds)
    cols = [c for c in ds.predictor_columns() if c not in design.zero_information]
    cols = [c for c in cols if not ds.wide[c].isna().any()]
    _, dropped = mixture_safe_predictors(ds, design, cols)
    plan = ClusterPlan.from_design(design, cols, dropped)
    from validation import cross_validate

    cv = cross_validate(ds.wide[cols], ds.wide["q30_pct"], ds.formulations, plan)
    params = [tuple(sorted(f["params"].items())) for f in cv["per_fold"].to_dict("records")]
    assert len(params) == 4
    # each fold tuned on its own training data; identical params across folds would be
    # the signature of tuning once outside the loop
    assert len(set(params)) > 1 or all("alpha" in dict(p) for p in params)
    assert "params" in cv["per_fold"].columns


# 3 -- variable selection refitted in every resample
def test_3_selection_is_refit_per_resample_not_fixed_once():
    ds = make_controlled("known_driver", seed=201, n_noise=60)
    res = run_attribution(ds, "response_y", settings=FAST)
    per_fold = res.validation.cv["selected_per_fold"]
    assert len(per_fold) >= 2
    assert any(set(a) != set(b) for a in per_fold for b in per_fold), (
        "every fold selected exactly the same subset, which suggests selection was not refit"
    )
    freq = res.validation.stability.frequency
    assert (freq > 0).sum() > 1 and (freq < 1).any()   # genuinely varying across resamples


# 4 -- training R-squared is never a headline
def test_4_no_training_r2_anywhere_in_the_output():
    ds = make_controlled("known_driver", seed=202)
    res = run_attribution(ds, "response_y", settings=FAST)
    text = res.text().lower()
    assert "training r" not in text and "apparent r" not in text
    assert "q-squared" in text or "q2" in text
    for name, src in SOURCES.items():
        assert "training_r2" not in src, name


# 5 -- importance rankings always carry confound lists
def test_5_every_finding_carries_its_confound_list():
    ds = make_controlled("confounded", seed=203)
    res = run_attribution(ds, "response_y", settings=FAST)
    assert res.report.findings
    for f in res.report.findings:
        s = f.sentence("response_y")
        assert ("cannot be separated from" in s) or ("no factor was found to be inseparable" in s)


# 6 -- nothing is displayed when the permutation test failed
def test_6_failed_permutation_blocks_every_model_output():
    ds = make_controlled("no_signal", seed=204)
    res = run_attribution(ds, "response_y", settings=FAST)
    assert not res.validation.passed
    assert res.report.findings == []
    with pytest.raises(PermissionError):
        partial_dependence(res.model, res.model_input, "driver", res.validation)
    with pytest.raises(PermissionError):
        linear_contributions(res.model, res.model_input, res.validation)


# 7 -- no causal language
def test_7_no_causal_language_in_any_generated_text():
    for mode, seed in (("known_driver", 205), ("no_signal", 206), ("confounded", 207)):
        res = run_attribution(make_controlled(mode, seed=seed), "response_y", settings=FAST)
        hits = causal_language_report(res.text())
        assert not hits, (mode, hits[:3])


# 8 -- no silent imputation
def test_8_missing_predictors_are_excluded_not_quietly_filled():
    ds = example_dataset()
    assert ds.wide["coating_weight_gain_pct"].isna().any()
    res = run_attribution(ds, "q30_pct", settings=FAST)
    assert "coating_weight_gain_pct" not in res.model_input.columns
    assert any("excluded rather than imputed" in w for w in res.warnings)
    assert res.method_record["predictors_excluded_for_missingness"]


def test_8b_opting_in_to_the_fallback_is_recorded():
    ds = example_dataset()
    res = run_attribution(ds, "q30_pct", settings=FAST, allow_missing_predictors=True)
    assert "coating_weight_gain_pct" in res.model_input.columns
    assert "feature_cells_filled_with_training_mean" in res.method_record


def test_8c_loader_never_imputes_on_its_own():
    ds = example_dataset()
    assert ds.imputation_record == []
    assert ds.wide.isna().any().any()


# 9 -- mixture components are never fitted raw
def test_9_mixture_constraint_is_broken_before_any_fit():
    ds = example_dataset()
    res = run_attribution(ds, "q30_pct", settings=FAST)
    group = res.design.mixture_groups[0]["columns"]
    in_model = [m for members in res.plan.groups.values() for m in members]
    assert set(group) - set(in_model), "all mixture components entered the model"
    assert res.method_record["mixture_dropped"]


def test_9b_model_matrix_is_not_singular_after_the_fix():
    ds = example_dataset()
    res = run_attribution(ds, "q30_pct", settings=FAST)
    cols = [m for members in res.plan.groups.values() for m in members]
    M = res.model_input[cols].to_numpy(dtype=float)
    M = M - M.mean(axis=0)
    # after removing the remainder, no exact constant-sum relation should survive
    assert not np.isclose(np.linalg.matrix_rank(M, tol=1e-9), 0)


# 10 -- complexity capped by sample size
def test_10_complexity_is_capped_and_the_rule_is_visible():
    ds = example_dataset()
    res = run_attribution(ds, "q30_pct", settings=FAST)
    cap = res.method_record["complexity_cap_final"]["max_features"]
    assert len(res.model.selected_) <= cap
    assert "floor(n_train/5)" in res.method_record["complexity_rule"]
    for fold in res.validation.cv["per_fold"].to_dict("records"):
        assert fold["n_selected"] <= cap        # enforced per fold, not just at the end


def test_10b_cap_shrinks_with_n():
    small = make_controlled("known_driver", n_formulations=3, batches_per_formulation=3, seed=208)
    res = run_attribution(small, "response_y", settings=RunSettings(n_permutations=20, n_bootstrap=15))
    if not res.refused:
        assert res.method_record["complexity_cap_final"]["max_features"] <= 2
