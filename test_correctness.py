"""Prompt 8 -- CORRECTNESS suite.

The most important test in this file is the one that asserts the app finds
nothing in noise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import RunSettings
from diagnostics import assess_design, mixture_safe_predictors
from interpret import causal_language_report, partial_dependence
from loader import build_dataset, long_to_wide
from modelling import ClusterPlan, univariate_screen
from pipeline import run_attribution
from response import (
    dissolution_efficiency,
    f2_similarity,
    fit_weibull,
    mean_dissolution_time,
    time_to_percent,
)
from synthetic import GROUND_TRUTH, example_dataset, make_controlled, make_synthetic
from validation import cross_validate, make_splits, random_split_q2

FAST = RunSettings(n_permutations=60, n_bootstrap=40, inner_cv_splits=4)


# ---------------------------------------------------------------------
# 1. known driver
# ---------------------------------------------------------------------
def test_recovers_a_known_driver():
    ds = make_controlled("known_driver", seed=11)
    res = run_attribution(ds, "response_y", settings=FAST)
    assert not res.refused
    assert res.validation.permutation["passed"], res.validation.text()
    assert res.validation.cv["q2"] > 0.5
    top = res.report.findings[0]
    assert "driver" in top.members, [f.feature for f in res.report.findings]
    assert top.selection_frequency > 0.9


def test_known_driver_effect_direction_is_right():
    ds = make_controlled("known_driver", seed=12, effect=4.0)
    res = run_attribution(ds, "response_y", settings=FAST)
    top = [f for f in res.report.findings if "driver" in f.members][0]
    assert top.coefficient_std > 0                       # positive effect recovered
    assert top.ci_low > 0                                # interval excludes zero


# ---------------------------------------------------------------------
# 2. no signal -- the important one
# ---------------------------------------------------------------------
def test_pure_noise_is_refused():
    ds = make_controlled("no_signal", seed=21)
    res = run_attribution(ds, "response_y", settings=FAST)
    assert not res.validation.permutation["passed"], res.validation.text()
    assert res.validation.verdict == "not_supported"
    assert res.report.findings == []
    assert "no attribution supported" in res.report.headline.lower()


def test_no_signal_blocks_dependence_plots():
    ds = make_controlled("no_signal", seed=22)
    res = run_attribution(ds, "response_y", settings=FAST)
    with pytest.raises(PermissionError):
        partial_dependence(res.model, ds.wide[ds.predictor_columns()], "driver", res.validation)


@pytest.mark.parametrize("seed", [31, 32, 33])
def test_noise_refusal_is_not_a_fluke(seed):
    ds = make_controlled("no_signal", seed=seed)
    res = run_attribution(ds, "response_y", settings=FAST)
    assert res.validation.verdict == "not_supported"


# ---------------------------------------------------------------------
# 3. confounded pair
# ---------------------------------------------------------------------
def test_confounded_pair_reported_as_inseparable():
    ds = make_controlled("confounded", seed=41)
    design = assess_design(ds)
    groups = [set(g) for g in design.inseparable_groups]
    assert {"driver", "partner"} in groups, design.inseparable_groups
    txt = design.text().lower()
    assert "cannot be separated" in txt


def test_confounded_pair_is_not_split_by_the_model():
    ds = make_controlled("confounded", seed=42)
    res = run_attribution(ds, "response_y", settings=FAST)
    feature_members = {f.feature: set(f.members) for f in res.report.findings}
    hit = [m for m in feature_members.values() if "driver" in m]
    assert hit, feature_members
    assert hit[0] == {"driver", "partner"}          # reported as the pair, not one of them
    top = res.report.findings[0]
    assert top.is_cluster
    assert "cluster" in top.feature


def test_example_dataset_confounding_matches_ground_truth():
    ds = example_dataset()
    design = assess_design(ds)
    truth = set(GROUND_TRUTH["inseparable_cluster"])
    found = [set(g) for g in design.inseparable_groups]
    assert any(truth.issubset(g) for g in found), design.inseparable_groups


# ---------------------------------------------------------------------
# 4. grouped cross-validation really is grouped
# ---------------------------------------------------------------------
def test_grouped_cv_holds_out_whole_formulations():
    ds = example_dataset()
    groups = ds.formulations
    for tr, te in make_splits(groups, "leave_one_formulation_out"):
        assert set(tr).isdisjoint(set(te))                       # no batch in both
        assert set(groups.iloc[tr]).isdisjoint(set(groups.iloc[te]))   # no formulation in both


def test_leave_one_batch_out_holds_out_single_batches():
    ds = example_dataset()
    splits = make_splits(ds.formulations, "leave_one_batch_out")
    assert len(splits) == ds.n_batches
    for tr, te in splits:
        assert len(te) == 1
        assert set(tr).isdisjoint(set(te))


def test_no_random_split_is_reachable_from_make_splits():
    with pytest.raises(ValueError):
        make_splits(pd.Series(["F1"] * 6), "random")


# ---------------------------------------------------------------------
# 5. random splitting inflates performance, grouped is reported
# ---------------------------------------------------------------------
def test_random_split_inflates_q2_and_grouped_is_reported():
    ds = example_dataset()
    design = assess_design(ds)
    cols = [c for c in ds.predictor_columns() if c not in design.zero_information]
    _, dropped = mixture_safe_predictors(ds, design, cols)
    plan = ClusterPlan.from_design(design, cols, dropped)
    X, y, g = ds.wide[cols], ds.wide["q30_pct"], ds.formulations
    grouped = cross_validate(X, y, g, plan)["q2"]
    random = random_split_q2(X, y, plan)
    assert random > grouped                       # the leak is real and measurable
    res = run_attribution(ds, "q30_pct", settings=FAST)
    assert res.validation.cv["q2"] == pytest.approx(grouped, abs=1e-9)
    assert f"{grouped:.3f}" in res.validation.text()


# ---------------------------------------------------------------------
# loader validation
# ---------------------------------------------------------------------
def _tiny(wide=None, meta=None, families=None):
    wide = wide if wide is not None else pd.DataFrame(
        {"batch": ["A", "B", "C", "D"], "x": [1.0, 2.0, 3.0, 4.0], "y": [1.0, 2.0, 3.0, 5.0]}
    )
    meta = meta if meta is not None else pd.DataFrame(
        {"batch": ["A", "B", "C", "D"], "formulation": ["F1", "F1", "F2", "F2"]}
    )
    families = families if families is not None else {"x": "PROCESS", "y": "RESPONSE"}
    return build_dataset(wide, families, meta)


def test_duplicate_batch_ids_are_an_error():
    wide = pd.DataFrame({"batch": ["A", "A", "B", "C"], "x": [1.0, 2, 3, 4], "y": [1.0, 2, 3, 4]})
    meta = pd.DataFrame({"batch": ["A", "B", "C"], "formulation": ["F1", "F1", "F2"]})
    ds = _tiny(wide, meta)
    assert ds.report.has("DUPLICATE_BATCH_ID")
    assert not ds.report.ok


def test_constant_columns_are_excluded_with_a_reason():
    wide = pd.DataFrame(
        {"batch": ["A", "B", "C", "D"], "x": [5.0] * 4, "z": [1.0, 2, 3, 4], "y": [1.0, 2, 3, 4]}
    )
    ds = _tiny(wide, families={"x": "PROCESS", "z": "PROCESS", "y": "RESPONSE"})
    assert ds.report.has("CONSTANT_COLUMN")
    assert "x" in ds.excluded and "constant" in ds.excluded["x"]
    assert "x" not in ds.predictor_columns()


def test_high_missingness_is_reported_and_not_imputed():
    wide = pd.DataFrame(
        {"batch": ["A", "B", "C", "D"], "x": [1.0, np.nan, np.nan, np.nan],
         "z": [1.0, 2, 3, 4], "y": [1.0, 2, 3, 4]}
    )
    ds = _tiny(wide, families={"x": "PROCESS", "z": "PROCESS", "y": "RESPONSE"})
    assert ds.report.has("HIGH_MISSINGNESS")
    assert ds.wide["x"].isna().sum() == 3               # untouched
    assert ds.imputation_record == []


def test_imputation_is_explicit_and_recorded():
    wide = pd.DataFrame(
        {"batch": ["A", "B", "C", "D"], "x": [1.0, np.nan, 3.0, 4.0], "y": [1.0, 2, 3, 4]}
    )
    ds = _tiny(wide)
    ds.impute("median", ["x"])
    assert ds.wide["x"].isna().sum() == 0
    assert ds.imputation_record[0]["method"] == "median"
    assert ds.imputation_record[0]["column"] == "x"


def test_non_numeric_contamination_is_not_coerced():
    wide = pd.DataFrame(
        {"batch": ["A", "B", "C", "D"], "x": [1.0, 2.0, "n/a", 4.0], "y": [1.0, 2, 3, 4]}
    )
    ds = _tiny(wide)
    assert ds.report.has("NON_NUMERIC_IN_NUMERIC_COLUMN")
    assert "x" in ds.excluded
    assert ds.wide["x"].tolist()[2] == "n/a"            # value preserved, not turned into NaN


def test_batches_missing_from_metadata_are_flagged():
    wide = pd.DataFrame({"batch": ["A", "B", "C"], "x": [1.0, 2, 3], "y": [1.0, 2, 3]})
    meta = pd.DataFrame({"batch": ["A", "B"], "formulation": ["F1", "F2"]})
    ds = _tiny(wide, meta)
    assert ds.report.has("BATCH_MISSING_METADATA")


def test_long_format_pivot_and_duplicate_refusal():
    long = pd.DataFrame(
        {"batch": ["A", "A", "B", "B"], "variable": ["x", "y", "x", "y"], "value": [1, 2, 3, 4]}
    )
    wide = long_to_wide(long)
    assert set(wide.columns) == {"batch", "x", "y"}
    dupe = pd.concat([long, long.iloc[[0]]])
    with pytest.raises(ValueError):
        long_to_wide(dupe)


# ---------------------------------------------------------------------
# mixture constraint
# ---------------------------------------------------------------------
def test_mixture_constraint_detected_and_broken_before_fitting():
    ds = example_dataset()
    design = assess_design(ds)
    assert design.mixture_groups, "mixture constraint not detected"
    cols = [c for c in ds.predictor_columns() if c not in design.zero_information]
    kept, dropped = mixture_safe_predictors(ds, design, cols)
    assert dropped
    sums = ds.wide[design.mixture_groups[0]["columns"]].sum(axis=1)
    assert sums.std() < 0.5
    plan = ClusterPlan.from_design(design, cols, dropped)
    flat = [m for members in plan.groups.values() for m in members]
    assert not set(dropped) & set(flat)


def test_singular_mixture_is_not_fitted_raw():
    """All components in one linear model would be singular; the plan must drop one."""
    ds = example_dataset()
    design = assess_design(ds)
    cols = [c for c in ds.predictor_columns() if c not in design.zero_information]
    _, dropped = mixture_safe_predictors(ds, design, cols)
    group = design.mixture_groups[0]["columns"]
    assert set(group) - set(dropped)          # some retained
    assert set(group) & set(dropped)          # at least one removed


# ---------------------------------------------------------------------
# profile metrics
# ---------------------------------------------------------------------
def test_profile_metrics_against_analytic_values():
    t = np.array([0.25, 0.5, 1, 2, 4, 6, 8, 12], dtype=float)
    td, beta = 3.0, 1.0
    y = 100 * (1 - np.exp(-((t / td) ** beta)))
    fit = fit_weibull(t, y)
    assert fit["weibull_td"] == pytest.approx(td, rel=0.05)
    assert fit["weibull_beta"] == pytest.approx(beta, rel=0.05)
    assert time_to_percent(t, y, 50) == pytest.approx(2.08, abs=0.3)
    assert np.isnan(time_to_percent(t, np.linspace(1, 40, 8), 80))   # censored, not extrapolated
    assert 0 < dissolution_efficiency(t, y) < 100
    assert mean_dissolution_time(t, y) > 0


def test_f2_identity_and_difference():
    r = np.array([20.0, 40, 60, 80, 90])
    assert f2_similarity(r, r) == pytest.approx(100.0)
    assert f2_similarity(r, r - 20) < 50


def test_response_refuses_when_range_is_within_method_noise():
    from response import build_response

    ds = example_dataset()
    r = build_response(ds, "assay_pct", method_rsd_pct=2.0)
    assert not r.usable
    assert "measurement noise" in r.text()


def test_response_reports_noise_swamping_signal():
    from response import build_response

    ds = example_dataset()
    r = build_response(ds, "content_uniformity_av")
    vc = r.diagnostics["variance_components"]
    assert np.isfinite(vc["f_ratio"])


# ---------------------------------------------------------------------
# screen, language, determinism
# ---------------------------------------------------------------------
def test_univariate_screen_is_labelled_a_screen():
    ds = make_controlled("known_driver", seed=51)
    scr = univariate_screen(ds.predictors(), ds.wide["response_y"])
    assert "q_value_bh" in scr.columns
    assert (scr["q_value_bh"].dropna() >= scr["p_value"].dropna() - 1e-12).all()
    assert "not findings" in scr.attrs["caption"]


def test_report_contains_no_causal_language():
    for mode, seed in (("known_driver", 61), ("no_signal", 62), ("confounded", 63)):
        ds = make_controlled(mode, seed=seed)
        res = run_attribution(ds, "response_y", settings=FAST)
        text = res.text()
        assert not causal_language_report(text), (mode, causal_language_report(text)[:3])


def test_findings_always_carry_a_confound_list_field():
    ds = make_controlled("confounded", seed=71)
    res = run_attribution(ds, "response_y", settings=FAST)
    for f in res.report.findings:
        assert isinstance(f.confounds, list)
        assert f.sentence("response_y")


def test_banner_present_on_findings():
    ds = make_controlled("known_driver", seed=72)
    res = run_attribution(ds, "response_y", settings=FAST)
    assert "Not causation" in res.report.banner
    assert res.report.banner in res.report.text()


def test_determinism_same_input_same_output():
    ds1 = make_controlled("known_driver", seed=81)
    ds2 = make_controlled("known_driver", seed=81)
    a = run_attribution(ds1, "response_y", settings=FAST)
    b = run_attribution(ds2, "response_y", settings=FAST)
    assert a.validation.cv["q2"] == b.validation.cv["q2"]
    assert a.validation.permutation["p_value"] == b.validation.permutation["p_value"]
    assert a.report.headline == b.report.headline
    assert list(a.validation.stability.frequency) == list(b.validation.stability.frequency)


def test_synthetic_example_is_reproducible():
    a = make_synthetic(seed=7)[0]
    b = make_synthetic(seed=7)[0]
    pd.testing.assert_frame_equal(a, b)


def test_complexity_cap_scales_with_n():
    from modelling import complexity_cap

    assert complexity_cap(10)["max_features"] == 2
    assert complexity_cap(24)["max_features"] == 4
    assert complexity_cap(3)["max_features"] == 1
    assert complexity_cap(500)["max_features"] == 10


def test_model_respects_the_cap():
    ds = example_dataset()
    design = assess_design(ds)
    cols = [c for c in ds.predictor_columns() if c not in design.zero_information]
    _, dropped = mixture_safe_predictors(ds, design, cols)
    plan = ClusterPlan.from_design(design, cols, dropped)
    from modelling import AttributionModel

    m = AttributionModel("elastic_net", plan).fit(ds.wide[cols], ds.wide["q30_pct"], ds.formulations)
    assert len(m.selected_) <= m.cap_["max_features"]
