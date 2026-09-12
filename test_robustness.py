"""Prompt 8 -- ROBUSTNESS suite.

Every method against every degenerate input. Crashes and silent wrong answers
must both be zero. Every refusal is audited to confirm the method genuinely
could not answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List

import numpy as np
import pandas as pd
import pytest

from config import RunSettings
from diagnostics import assess_design, mixture_safe_predictors
from errors import Refusal
from loader import build_dataset
from modelling import AttributionModel, ClusterPlan, stability_selection, univariate_screen
from pipeline import run_attribution
from response import build_response
from validation import cross_validate, permutation_test

FAST = RunSettings(n_permutations=20, n_bootstrap=15, inner_cv_splits=3)

OK, REFUSED, CRASH, SILENT_WRONG = "OK", "REFUSED", "CRASH", "SILENT WRONG ANSWER"


# ---------------------------------------------------------------------
# scenarios
# ---------------------------------------------------------------------
def _mk(wide: pd.DataFrame, meta: pd.DataFrame, families: Dict[str, str]):
    return build_dataset(wide, families, meta)


def _base(n=12, n_form=3, seed=5):
    rng = np.random.default_rng(seed)
    batch = [f"B{i:02d}" for i in range(n)]
    per = max(1, n // n_form)
    form = [f"F{min(i // per, n_form - 1) + 1}" for i in range(n)]
    wide = pd.DataFrame(
        {
            "batch": batch,
            "x1": rng.normal(0, 1, n),
            "x2": rng.normal(5, 2, n),
            "x3": rng.normal(-3, 1, n),
            "y": rng.normal(50, 5, n),
        }
    )
    meta = pd.DataFrame(
        {
            "batch": batch,
            "formulation": form,
            "manufacturing_date": pd.date_range("2024-01-01", periods=n, freq="7D"),
        }
    )
    families = {"x1": "PROCESS", "x2": "PROCESS", "x3": "PROCESS", "y": "RESPONSE"}
    return wide, meta, families


def scenario(name: str):
    wide, meta, fam = _base()
    if name == "nominal":
        pass
    elif name == "all_identical_response":
        wide["y"] = 42.0
    elif name == "negative_values":
        wide["y"] = -np.abs(wide["y"])
        wide["x1"] = -np.abs(wide["x1"])
    elif name == "zeros":
        wide[["x1", "x2", "x3"]] = 0.0
        wide["y"] = 0.0
    elif name == "single_batch":
        wide, meta = wide.iloc[:1].copy(), meta.iloc[:1].copy()
    elif name == "two_batches":
        wide, meta = wide.iloc[:2].copy(), meta.iloc[:2].copy()
    elif name == "one_formulation":
        meta["formulation"] = "F1"
    elif name == "more_predictors_than_batches":
        rng = np.random.default_rng(3)
        wide = wide.iloc[:6].copy()
        meta = meta.iloc[:6].copy()
        for j in range(30):
            wide[f"p{j}"] = rng.normal(0, 1, 6)
            fam[f"p{j}"] = "PROCESS"
    elif name == "all_missing_column":
        wide["x2"] = np.nan
    elif name == "constant_column":
        wide["x2"] = 7.0
    elif name == "duplicate_batch_ids":
        wide.loc[1, "batch"] = wide.loc[0, "batch"]
    elif name == "non_numeric_contamination":
        wide["x1"] = wide["x1"].astype(object)
        wide.loc[2, "x1"] = "not measured"
    elif name == "missing_response_values":
        wide.loc[[0, 1, 2], "y"] = np.nan
    elif name == "response_with_one_distinct_value_plus_nan":
        wide["y"] = np.nan
        wide.loc[0, "y"] = 3.0
    else:
        raise KeyError(name)
    return wide, meta, fam


SCENARIOS = [
    "nominal",
    "all_identical_response",
    "negative_values",
    "zeros",
    "single_batch",
    "two_batches",
    "one_formulation",
    "more_predictors_than_batches",
    "all_missing_column",
    "constant_column",
    "duplicate_batch_ids",
    "non_numeric_contamination",
    "missing_response_values",
    "response_with_one_distinct_value_plus_nan",
]


# ---------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------
@dataclass
class Outcome:
    scenario: str
    method: str
    status: str
    detail: str = ""


def _classify(fn: Callable, check=None) -> Outcome:
    try:
        out = fn()
    except Refusal as exc:
        return Outcome("", "", REFUSED, str(exc)[:160])
    except (ValueError, KeyError) as exc:
        # a stated, typed data complaint is a refusal; anything else is a crash
        return Outcome("", "", REFUSED, f"{type(exc).__name__}: {exc}"[:160])
    except Exception as exc:  # noqa: BLE001
        return Outcome("", "", CRASH, f"{type(exc).__name__}: {exc}"[:200])
    if check is not None:
        problem = check(out)
        if problem:
            return Outcome("", "", SILENT_WRONG, problem)
    return Outcome("", "", OK, "")


def _prepare(name):
    wide, meta, fam = scenario(name)
    ds = _mk(wide, meta, fam)
    design = assess_design(ds)
    cols = [c for c in ds.predictor_columns() if c not in design.zero_information]
    _, dropped = mixture_safe_predictors(ds, design, cols)
    plan = ClusterPlan.from_design(design, cols, dropped)
    return ds, design, cols, plan


# ---- per-method silent-wrong checks ---------------------------------
def _check_design(name):
    def check(design):
        if name == "constant_column" and "x2" not in design.zero_information:
            return "constant column not identified as carrying no information"
        if name == "all_missing_column" and "x2" in design.range_table.index and design.range_table.loc["x2", "n_distinct"] > 0:
            return "all-missing column reported as having distinct values"
        return ""
    return check


def _check_response(name):
    def check(resp):
        if name in ("all_identical_response", "zeros") and resp.usable:
            return "a response with no variation was accepted as modellable"
        if name == "response_with_one_distinct_value_plus_nan" and resp.usable:
            return "a response with one observation was accepted as modellable"
        return ""
    return check


def _check_pipeline(name):
    def check(res):
        if res.refused:
            return ""
        v = res.validation
        if name in ("all_identical_response", "zeros", "single_batch", "two_batches"):
            return f"degenerate input produced a verdict of {v.verdict!r}"
        if name == "one_formulation":
            return "one formulation produced a validated attribution"
        if v.verdict != "not_supported" and not v.permutation.get("passed"):
            return "verdict claims support while the permutation test failed"
        if res.report and res.report.findings and not v.passed:
            return "findings displayed although validation did not pass"
        for f in (res.report.findings if res.report else []):
            if any(m in res.design.zero_information for m in f.members):
                return "a zero-information column appears in the findings"
            if any(m in ("x1",) and name == "non_numeric_contamination" for m in f.members):
                return "a contaminated column was used as a predictor"
        if np.isfinite(v.cv["q2"]) and v.cv["q2"] > 0.99 and name == "nominal":
            return "implausibly perfect cross-validated fit on random data"
        return ""
    return check


# ---------------------------------------------------------------------
# the matrix
# ---------------------------------------------------------------------
def run_matrix() -> List[Outcome]:
    results: List[Outcome] = []

    def rec(scn, method, outcome: Outcome):
        outcome.scenario, outcome.method = scn, method
        results.append(outcome)

    for name in SCENARIOS:
        rec(name, "build_dataset", _classify(lambda n=name: _mk(*scenario(n))))
        try:
            ds, design, cols, plan = _prepare(name)
        except (Refusal, ValueError, KeyError) as exc:
            for m in ("assess_design", "build_response", "univariate_screen", "model_fit",
                      "cross_validate", "permutation_test", "stability_selection", "run_attribution"):
                rec(name, m, Outcome(name, m, REFUSED, str(exc)[:120]))
            continue
        except Exception as exc:  # noqa: BLE001
            rec(name, "prepare", Outcome(name, "prepare", CRASH, repr(exc)[:200]))
            continue

        rec(name, "assess_design", _classify(lambda n=name: _prepare(n)[1], _check_design(name)))
        rec(name, "build_response",
            _classify(lambda d=ds: build_response(d, "y"), _check_response(name)))

        X = ds.wide[cols]
        y = pd.to_numeric(ds.wide["y"], errors="coerce") if "y" in ds.wide else pd.Series(dtype=float)
        g = ds.formulations

        rec(name, "univariate_screen", _classify(lambda: univariate_screen(X, y)))
        rec(name, "model_fit",
            _classify(lambda: AttributionModel("elastic_net", plan).fit(X, y, g)))
        rec(name, "cross_validate", _classify(lambda: cross_validate(X, y, g, plan)))
        rec(name, "permutation_test",
            _classify(lambda: permutation_test(X, y, g, plan, n_permutations=10)))
        rec(name, "stability_selection",
            _classify(lambda: stability_selection(X, y, g, plan, n_boot=10)))
        rec(name, "run_attribution",
            _classify(lambda d=ds: run_attribution(d, "y", settings=FAST), _check_pipeline(name)))
    return results


@pytest.fixture(scope="module")
def matrix():
    return run_matrix()


def test_no_crashes(matrix):
    crashes = [o for o in matrix if o.status == CRASH]
    assert not crashes, "\n".join(f"{o.scenario}/{o.method}: {o.detail}" for o in crashes)


def test_no_silent_wrong_answers(matrix):
    wrong = [o for o in matrix if o.status == SILENT_WRONG]
    assert not wrong, "\n".join(f"{o.scenario}/{o.method}: {o.detail}" for o in wrong)


def test_every_refusal_states_a_reason(matrix):
    bare = [o for o in matrix if o.status == REFUSED and len(o.detail.strip()) < 15]
    assert not bare, [f"{o.scenario}/{o.method}" for o in bare]


def test_degenerate_scenarios_are_actually_refused(matrix):
    """Audit: these could not have been answered, so a non-refusal would be a bug."""
    must_refuse = {
        ("single_batch", "cross_validate"),
        ("single_batch", "model_fit"),
        ("two_batches", "cross_validate"),
        ("one_formulation", "cross_validate"),
        ("all_identical_response", "cross_validate"),
    }
    by_key = {(o.scenario, o.method): o for o in matrix}
    for key in must_refuse:
        assert key in by_key, key
        assert by_key[key].status == REFUSED, (key, by_key[key].status, by_key[key].detail)


def test_pipeline_refuses_degenerate_inputs_with_reasons(matrix):
    for scn in ("single_batch", "two_batches", "all_identical_response", "one_formulation"):
        o = [x for x in matrix if x.scenario == scn and x.method == "run_attribution"][0]
        assert o.status in (REFUSED, OK)
        if o.status == OK:
            res = run_attribution(_mk(*scenario(scn)), "y", settings=FAST)
            assert res.refused, f"{scn} was not refused"
            assert res.refusals and len(res.refusals[0]) > 20


def test_matrix_summary_is_printable(matrix, capsys):
    df = pd.DataFrame([o.__dict__ for o in matrix])
    counts = df["status"].value_counts().to_dict()
    print(df.pivot_table(index="scenario", columns="method", values="status", aggfunc="first"))
    print(counts)
    assert counts.get(CRASH, 0) == 0
    assert counts.get(SILENT_WRONG, 0) == 0
