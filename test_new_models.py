"""Group lasso and Scheffe mixture model.

Two additions to the ladder, each earning its place by doing something the existing
rungs could not: keeping an inseparable cluster together as a unit, and modelling the
simplex instead of deleting a component from it.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from config import RunSettings
from diagnostics import assess_design, mixture_safe_predictors
from errors import Refusal
from interpret import causal_language_report
from modelling import (
    AttributionModel,
    ClusterPlan,
    group_lasso_lambda_max,
    group_lasso_path,
    pseudocomponents,
    scheffe_terms,
)
from pipeline import run_attribution
from synthetic import example_dataset, make_controlled, make_mixture
from validation import cross_validate

FAST = RunSettings(n_permutations=40, n_bootstrap=30, inner_cv_splits=3)
SMALL = RunSettings(n_permutations=25, n_bootstrap=20, inner_cv_splits=3)


def _plan_for(ds, response):
    design = assess_design(ds)
    cols = [
        c for c in ds.predictor_columns()
        if c not in design.zero_information and not ds.wide[c].isna().any()
    ]
    _, dropped = mixture_safe_predictors(ds, design, cols)
    return design, cols, ClusterPlan.from_design(design, cols, dropped)


# ---------------------------------------------------------------------
# group lasso -- solver correctness first
# ---------------------------------------------------------------------
def test_solver_zeroes_everything_at_lambda_max():
    rng = np.random.default_rng(1)
    Z = rng.normal(size=(30, 6))
    Z = (Z - Z.mean(0)) / Z.std(0)
    y = Z[:, 0] * 2 + rng.normal(0, 0.5, 30)
    blocks = [np.array([0, 1]), np.array([2, 3]), np.array([4]), np.array([5])]
    lam_max = group_lasso_lambda_max(Z, y, blocks)
    coefs, _ = group_lasso_path(Z, y, blocks, np.array([lam_max * 1.001]))
    assert np.allclose(coefs[:, 0], 0.0, atol=1e-8)


def test_solver_approaches_least_squares_at_small_lambda():
    rng = np.random.default_rng(2)
    Z = rng.normal(size=(40, 4))
    Z = (Z - Z.mean(0)) / Z.std(0)
    y = Z @ np.array([2.0, -1.0, 0.5, 0.0]) + rng.normal(0, 0.2, 40)
    blocks = [np.array([i]) for i in range(4)]
    lam_max = group_lasso_lambda_max(Z, y, blocks)
    coefs, inter = group_lasso_path(Z, y, blocks, np.array([lam_max * 1e-6]))
    ols = np.linalg.lstsq(np.column_stack([np.ones(40), Z]), y, rcond=None)[0]
    assert coefs[:, 0] == pytest.approx(ols[1:], abs=0.02)
    assert inter[0] == pytest.approx(ols[0], abs=0.02)


def test_solver_is_deterministic():
    rng = np.random.default_rng(3)
    Z = rng.normal(size=(25, 8))
    y = rng.normal(size=25)
    blocks = [np.array([0, 1, 2]), np.array([3, 4]), np.array([5]), np.array([6, 7])]
    lam = np.logspace(-1, -3, 5)
    a, _ = group_lasso_path(Z, y, blocks, lam)
    b, _ = group_lasso_path(Z, y, blocks, lam)
    assert np.array_equal(a, b)


# ---------------------------------------------------------------------
# group lasso -- the property it exists for
# ---------------------------------------------------------------------
def test_group_lasso_keeps_an_inseparable_pair_together():
    """The whole point: never one member of a confounded pair without its twin."""
    ds = make_controlled("confounded", seed=301)
    design, cols, plan = _plan_for(ds, "response_y")
    m = AttributionModel("group_lasso", plan).fit(
        ds.wide[cols], ds.wide["response_y"], ds.formulations
    )
    pair = [f for f, mem in m.feature_members_.items() if set(mem) == {"driver", "partner"}]
    assert pair, m.feature_members_
    coefs = m._member_coefs_[pair[0]]
    nonzero = [abs(v) > 1e-10 for v in coefs.values()]
    assert all(nonzero) or not any(nonzero), coefs      # together, or not at all
    assert all(nonzero), "the true pair should have been selected"


def test_lasso_can_split_the_pair_but_group_lasso_never_does():
    """Why group lasso was added.

    Plain LASSO sometimes keeps both twins and sometimes keeps one, depending on the
    data draw; there is no guarantee. Group lasso guarantees the pair moves as a unit.
    The point is not that LASSO is always wrong, it is that it offers no assurance.
    """
    split_seen, group_split_seen = False, False
    for seed in range(302, 314):
        ds = make_controlled("confounded", seed=seed)
        design, cols, plan = _plan_for(ds, "response_y")
        flat = ClusterPlan.identity(cols)
        y, g = ds.wide["response_y"], ds.formulations

        lasso = AttributionModel("lasso", flat).fit(ds.wide[cols], y, g)
        if len({f for f in lasso.selected_ if f in ("driver", "partner")}) == 1:
            split_seen = True

        gl = AttributionModel("group_lasso", plan).fit(ds.wide[cols], y, g)
        pair = [f for f, mem in gl.feature_members_.items() if set(mem) == {"driver", "partner"}]
        if pair:
            vals = [abs(v) > 1e-10 for v in gl._member_coefs_[pair[0]].values()]
            if any(vals) and not all(vals):
                group_split_seen = True

    assert split_seen, "LASSO kept both twins in every draw; the contrast is not demonstrated"
    assert not group_split_seen, "group lasso split an inseparable pair, which it must never do"


def test_group_lasso_recovers_a_known_driver():
    ds = make_controlled("known_driver", seed=303)
    res = run_attribution(ds, "response_y", model_type="group_lasso", settings=FAST)
    assert res.validation.permutation["passed"], res.validation.text()
    top = res.report.findings[0]
    assert "driver" in top.members


def test_group_lasso_refuses_noise():
    ds = make_controlled("no_signal", seed=304)
    res = run_attribution(ds, "response_y", model_type="group_lasso", settings=FAST)
    assert res.validation.verdict == "not_supported"
    assert res.report.findings == []


def test_group_lasso_per_unit_effect_is_exact_not_averaged():
    ds = make_controlled("known_driver", seed=305, effect=4.0)
    design, cols, plan = _plan_for(ds, "response_y")
    X, y = ds.wide[cols], ds.wide["response_y"]
    m = AttributionModel("group_lasso", plan).fit(X, y, ds.formulations)
    eff = m.per_unit_effect("driver", "driver")
    assert np.isfinite(eff)
    assert eff == pytest.approx(4.0, rel=0.5)           # true slope is 4 per unit


def test_group_lasso_respects_the_cap():
    ds = example_dataset()
    design, cols, plan = _plan_for(ds, "q30_pct")
    m = AttributionModel("group_lasso", plan).fit(ds.wide[cols], ds.wide["q30_pct"], ds.formulations)
    assert len(m.selected_) <= m.cap_["max_features"]


def test_group_lasso_works_inside_grouped_cv():
    ds = make_controlled("known_driver", seed=306)
    design, cols, plan = _plan_for(ds, "response_y")
    cv = cross_validate(ds.wide[cols], ds.wide["response_y"], ds.formulations, plan,
                        model_type="group_lasso")
    assert cv["q2"] > 0.4
    assert len(cv["per_fold"]) >= 2


# ---------------------------------------------------------------------
# Scheffe mixture model
# ---------------------------------------------------------------------
def test_pseudocomponents_sum_to_one():
    ds = make_mixture(seed=311)
    comps = [c for c in ds.wide.columns if c.startswith("comp_")]
    P = pseudocomponents(ds.wide, comps)
    assert np.allclose(P.sum(axis=1), 1.0)


def test_scheffe_terms_shapes():
    P = np.array([[0.5, 0.3, 0.2], [0.2, 0.5, 0.3]])
    T1, n1 = scheffe_terms(P, ["a", "b", "c"], 1)
    T2, n2 = scheffe_terms(P, ["a", "b", "c"], 2)
    assert T1.shape == (2, 3) and n1 == ["a", "b", "c"]
    assert T2.shape == (2, 6) and "a x b" in n2


def test_scheffe_recovers_the_component_ranking():
    ds = make_mixture(seed=312)
    design, cols, plan = _plan_for(ds, "response_y")
    m = AttributionModel("scheffe_linear", plan).fit(
        ds.wide[cols], ds.wide["response_y"], ds.formulations
    )
    fitted = dict(zip(m.feature_names_, m.coef_))
    truth = ds.ground_truth_betas
    order_fit = [k for k, _ in sorted(fitted.items(), key=lambda kv: -kv[1])]
    order_true = [k for k, _ in sorted(truth.items(), key=lambda kv: -kv[1])]
    assert order_fit == order_true, (fitted, truth)


def test_scheffe_coefficients_are_close_to_the_true_betas():
    ds = make_mixture(seed=313, noise_sd=0.8)
    design, cols, plan = _plan_for(ds, "response_y")
    m = AttributionModel("scheffe_linear", plan).fit(
        ds.wide[cols], ds.wide["response_y"], ds.formulations
    )
    for name, beta in ds.ground_truth_betas.items():
        got = dict(zip(m.feature_names_, m.coef_))[name] / 100.0
        assert got == pytest.approx(beta, abs=3.0), (name, got, beta)


def test_scheffe_has_no_intercept_and_is_not_singular():
    """An ordinary regression on all components with an intercept is singular; the
    canonical form is not, which is the whole reason it exists."""
    ds = make_mixture(seed=314)
    design, cols, plan = _plan_for(ds, "response_y")
    m = AttributionModel("scheffe_linear", plan).fit(
        ds.wide[cols], ds.wide["response_y"], ds.formulations
    )
    assert m.params_["intercept"].startswith("none")
    assert np.isfinite(m.coef_).all()
    comps = [c for c in cols if c.startswith("comp_")]
    with_intercept = np.column_stack([np.ones(len(ds.wide)), ds.wide[comps].to_numpy(float)])
    assert np.linalg.matrix_rank(with_intercept) < with_intercept.shape[1]   # singular


def test_scheffe_contrasts_have_the_right_sign():
    ds = make_mixture(seed=315)
    design, cols, plan = _plan_for(ds, "response_y")
    m = AttributionModel("scheffe_linear", plan).fit(
        ds.wide[cols], ds.wide["response_y"], ds.formulations
    )
    con = m.component_contrasts()
    row = con[(con["with"] == "comp_a_pct_ww") & (con["replacing"] == "comp_d_pct_ww")]
    assert len(row) == 1
    assert float(row["difference"].iloc[0]) > 0        # beta_a (40) exceeds beta_d (5)


def test_scheffe_refuses_when_there_is_no_mixture():
    ds = make_controlled("known_driver", seed=316)
    design, cols, plan = _plan_for(ds, "response_y")
    assert not plan.mixture_components
    with pytest.raises(Refusal):
        AttributionModel("scheffe_linear", plan).fit(
            ds.wide[cols], ds.wide["response_y"], ds.formulations
        )


def test_scheffe_validates_end_to_end():
    ds = make_mixture(seed=317)
    res = run_attribution(ds, "response_y", model_type="scheffe_linear", settings=SMALL)
    assert res.validation.permutation["passed"], res.validation.text()
    assert res.report.findings


def test_scheffe_report_uses_contrast_language_not_per_sd():
    ds = make_mixture(seed=318)
    res = run_attribution(ds, "response_y", model_type="scheffe_linear", settings=SMALL)
    s = res.report.findings[0].sentence("response_y")
    assert "at constant total" in s
    assert "1 SD of" not in s
    assert "contrast and not an effect of adding" in s


def test_scheffe_headline_declares_what_is_not_in_the_model():
    ds = example_dataset()
    res = run_attribution(ds, "q30_pct", model_type="scheffe_linear", settings=SMALL)
    if res.report.findings:
        h = res.report.headline
        assert "components only" in h
        assert "not in the model" in h or "were not in the model" in h
        # the confound list still comes from the design, not from the model
        assert "main_compression_kN" in res.report.findings[0].confounds


def test_scheffe_warns_that_process_factors_are_absent():
    ds = example_dataset()
    res = run_attribution(ds, "q30_pct", model_type="scheffe_linear", settings=SMALL)
    assert any("mixture components only" in w for w in res.warnings)


# ---------------------------------------------------------------------
# dense models now have meaningful stability
# ---------------------------------------------------------------------
@pytest.mark.parametrize("model_type", ["ridge", "pls"])
def test_dense_models_do_not_report_every_feature_as_selected(model_type):
    ds = make_controlled("known_driver", seed=321)
    res = run_attribution(ds, "response_y", model_type=model_type, settings=SMALL)
    freq = res.validation.stability.frequency
    assert (freq < 0.999).any(), "every feature was 'selected', which would carry no information"
    assert res.validation.stability.n_effective > 0


def test_dense_selection_label_is_honest():
    ds = make_controlled("known_driver", seed=322)
    res = run_attribution(ds, "response_y", model_type="ridge", settings=SMALL)
    if res.report.findings:
        assert "top" in res.report.findings[0].selection_label


def test_new_models_produce_no_causal_language():
    for mt in ("group_lasso", "scheffe_linear"):
        ds = make_mixture(seed=331) if mt == "scheffe_linear" else make_controlled("known_driver", seed=331)
        res = run_attribution(ds, "response_y", model_type=mt, settings=SMALL)
        assert not causal_language_report(res.text())
