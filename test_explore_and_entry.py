"""Manual data entry, the analysis catalogue, and the analyses that are valid at
small n.

The load-bearing test here is the last group: that attribution stays unavailable
below ten batches no matter how the data got in, while the describe-and-compare
analyses stay available all the way down to two.
"""
from __future__ import annotations

import io

import numpy as np
import pandas as pd
import pytest

from catalogue import (
    ATTRIBUTION,
    BY_ID,
    CATALOGUE,
    DESCRIPTIVE,
    available_analyses,
    small_batch_guidance,
)
from errors import DataError, Refusal
from explore import (
    compare_batches,
    compare_profiles,
    descriptive_summary,
    f1_difference,
    profile_summary,
    spec_check,
    trend_table,
    what_changed,
)
from gating import ATTRIBUTION_IDS, QUICK_IDS, SessionState, tab_status
from loader import blank_entry_table, build_dataset, split_single_table
from synthetic import example_dataset


# ---------------------------------------------------------------------
# manual entry / paste
# ---------------------------------------------------------------------
def _typed(n=3):
    return pd.DataFrame(
        {
            "batch": [f"B{i+1:03d}" for i in range(n)],
            "formulation": ["F1", "F1", "F2"][:n],
            "hpmc_level_pct": [18.0, 18.4, 22.1][:n],
            "hardness_N": [55.0, 58.0, 80.0][:n],
            "q30_pct": [30.0, 28.5, 19.0][:n],
        }
    )


def test_blank_entry_table_has_the_required_columns():
    t = blank_entry_table(4, ["hardness_N"])
    assert list(t.columns) == ["batch", "formulation", "hardness_N"]
    assert len(t) == 4
    assert t["batch"].tolist() == ["B001", "B002", "B003", "B004"]


def test_single_table_splits_into_data_and_metadata():
    df = _typed()
    wide, meta = split_single_table(df)
    assert "formulation" not in wide.columns
    assert list(meta.columns) == ["batch", "formulation"]
    assert "hpmc_level_pct" in wide.columns


def test_single_table_without_formulation_is_refused_with_a_reason():
    with pytest.raises(DataError) as exc:
        split_single_table(_typed().drop(columns=["formulation"]))
    assert "formulation" in str(exc.value)
    assert "leave-one-formulation-out" in str(exc.value)


def test_single_table_without_batch_is_refused():
    with pytest.raises(DataError):
        split_single_table(_typed().drop(columns=["batch"]))


def test_typed_data_builds_a_working_dataset():
    wide, meta = split_single_table(_typed())
    ds = build_dataset(
        wide,
        {"hpmc_level_pct": "FORMULATION", "hardness_N": "IPQC_PHYSICAL", "q30_pct": "RESPONSE"},
        meta,
    )
    assert ds.n_batches == 3
    assert ds.n_formulations == 2
    assert ds.report.ok


def test_pasted_tab_separated_text_is_read():
    text = "batch\tformulation\thpmc_level_pct\nB001\tF1\t18.2\nB002\tF2\t22.4"
    df = pd.read_csv(io.StringIO(text), sep=None, engine="python")
    wide, meta = split_single_table(df)
    ds = build_dataset(wide, {"hpmc_level_pct": "FORMULATION"}, meta)
    assert ds.n_batches == 2


# ---------------------------------------------------------------------
# the catalogue: what is offered at what size
# ---------------------------------------------------------------------
def _tiny_ds(n_batches, n_formulations):
    rng = np.random.default_rng(0)
    forms = [f"F{(i % n_formulations) + 1}" for i in range(n_batches)]
    df = pd.DataFrame(
        {
            "batch": [f"B{i:03d}" for i in range(n_batches)],
            "formulation": forms,
            "x1": rng.normal(0, 1, n_batches),
            "y": rng.normal(50, 5, n_batches),
        }
    )
    wide, meta = split_single_table(df)
    return build_dataset(wide, {"x1": "PROCESS", "y": "RESPONSE"}, meta)


def test_two_batches_offer_comparison_but_never_attribution():
    ds = _tiny_ds(2, 2)
    avail = {r["analysis"].id: r["available"] for r in available_analyses(ds)}
    assert avail["compare"] and avail["what_changed"] and avail["descriptive"]
    assert not avail["attribution"], "attribution must not be offered at two batches"
    assert not avail["screen"]


def test_three_batches_still_refuse_attribution():
    ds = _tiny_ds(3, 2)
    avail = {r["analysis"].id: r["available"] for r in available_analyses(ds)}
    assert avail["what_changed"] and avail["trend"] and avail["design_check"]
    assert not avail["attribution"]


def test_attribution_unlocks_at_ten_batches_and_two_formulations():
    assert not {r["analysis"].id: r["available"] for r in available_analyses(_tiny_ds(9, 2))}["attribution"]
    assert not {r["analysis"].id: r["available"] for r in available_analyses(_tiny_ds(12, 1))}["attribution"]
    assert {r["analysis"].id: r["available"] for r in available_analyses(_tiny_ds(12, 2))}["attribution"]


def test_unavailable_analyses_say_why():
    for row in available_analyses(_tiny_ds(2, 1)):
        if not row["available"]:
            assert len(row["reason"]) > 10, row["analysis"].id
            assert "needs" in row["reason"]


def test_every_catalogue_entry_is_described_in_plain_terms():
    for a in CATALOGUE:
        assert a.name and not a.name.endswith(".")
        assert a.question.endswith("?")
        assert a.group in (DESCRIPTIVE, ATTRIBUTION) or a.group == "Dissolution profiles"
        assert a.min_batches >= 1


def test_catalogue_ids_match_the_gating_sets():
    ids = {a.id for a in CATALOGUE}
    assert ATTRIBUTION_IDS.issubset(ids)
    assert QUICK_IDS.issubset(ids)
    assert not (ATTRIBUTION_IDS & QUICK_IDS)


def test_quick_analyses_unlock_their_own_tab_only():
    s = SessionState(data_loaded=True, analysis_id="what_changed", quick_ready=True)
    status = tab_status(s)
    assert status["Quick analysis"]["unlocked"]
    assert not status["Design diagnostics"]["unlocked"]
    assert not status["Findings"]["unlocked"]


def test_guidance_is_honest_about_small_n():
    assert "attribution" in small_batch_guidance(3, 1).lower()
    assert "confounded" in small_batch_guidance(3, 1)
    assert "one formulation" in small_batch_guidance(15, 1)
    assert "All analyses are available" in small_batch_guidance(30, 5)


# ---------------------------------------------------------------------
# the small-n analyses themselves
# ---------------------------------------------------------------------
def test_compare_batches_returns_one_column_per_batch():
    ds = example_dataset()
    out = compare_batches(ds, ["B001", "B002", "B003"])
    assert list(out.columns)[1:] == ["B001", "B002", "B003"]
    assert "family" in out.columns


def test_what_changed_works_with_only_two_batches():
    ds = _tiny_ds(2, 2)
    out = what_changed(ds, "B000", "B001")
    assert not out.empty
    assert {"difference", "percent_change"}.issubset(out.columns)
    assert out["sd_across_all_batches"].isna().all()      # no spread from two batches
    assert "can be credited" in out.attrs["caption"]


def test_what_changed_states_how_many_gaps_are_expected_by_chance():
    ds = example_dataset()
    out = what_changed(ds, "B001", "B024")
    cap = out.attrs["caption"]
    assert "by chance alone" in cap
    assert "deliberately changed" in cap


def test_what_changed_refuses_the_same_batch_twice():
    ds = example_dataset()
    with pytest.raises(Refusal):
        what_changed(ds, "B001", "B001")


def test_f2_comparison_needs_only_two_batches():
    ds = example_dataset()
    r = compare_profiles(ds, "B001", "B024", medium="0.1N HCl")
    assert np.isfinite(r["f2"]) and np.isfinite(r["f1"])
    assert r["n_points"] >= 3
    assert "similar" in r["verdict"]


def test_f2_flags_too_many_points_above_85_percent():
    ds = example_dataset()
    r = compare_profiles(ds, "B001", "B002", medium="0.1N HCl")
    if sum(1 for n in r["notes"] if "85%" in n):
        assert any("inflated" in n for n in r["notes"])


def test_f1_is_zero_for_identical_profiles():
    r = np.array([20.0, 40, 60, 80])
    assert f1_difference(r, r) == pytest.approx(0.0)
    assert f1_difference(r, r * 0.5) > 0


def test_profile_summary_covers_every_batch():
    ds = example_dataset()
    out = profile_summary(ds, "0.1N HCl")
    assert len(out) == ds.n_batches
    assert {"mdt", "t50", "dissolution_efficiency", "weibull_td"}.issubset(out.columns)


def test_trend_table_is_in_manufacturing_order_and_is_not_a_control_chart():
    ds = example_dataset()
    t = trend_table(ds, "hardness_N")
    dates = pd.to_datetime(t["manufacturing_date"])
    assert dates.is_monotonic_increasing
    assert "not control limits" in t.attrs["caption"]


def test_spec_check_classifies_each_batch():
    ds = example_dataset()
    out = spec_check(ds, {"assay_pct": (99.0, 101.0)})
    assert set(out["status"]) <= {"within limits", "ABOVE limit", "BELOW limit", "no result"}
    assert len(out) == ds.n_batches
    assert "not a release decision" in out.attrs["caption"]


def test_descriptive_summary_reports_rsd():
    ds = example_dataset()
    out = descriptive_summary(ds)
    assert {"n", "mean", "sd", "rsd_pct", "range"}.issubset(out.columns)
    assert (out["n"] > 0).all()


def test_small_n_analyses_never_fit_a_model():
    """None of these should be able to produce an attribution by any route."""
    import explore as ex

    src = (ex.__file__ or "")
    text = open(src).read()
    for banned in ("AttributionModel", "cross_validate", "permutation_test", "fit("):
        assert banned not in text, banned
