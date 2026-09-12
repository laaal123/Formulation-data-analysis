"""Prompt 7 -- interface and record tests.

The gating logic lives outside Streamlit precisely so that it can be tested here.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from config import RunSettings
from gating import FINDINGS_BANNER, TABS, SessionState, state_from_result, tab_status
from interpret import causal_language_report
from pipeline import run_attribution
from record import DECISION_RULES, REFERENCES, to_json, to_markdown
from synthetic import make_controlled

FAST = RunSettings(n_permutations=40, n_bootstrap=30, inner_cv_splits=3)
APP = Path(__file__).resolve().parent / "app.py"


# ---------------------------------------------------------------------
# gating
# ---------------------------------------------------------------------
def test_tab_order_is_the_analysis_order():
    assert TABS[0] == "Data"
    assert TABS.index("Choose analysis") == 1
    assert TABS.index("Design diagnostics") < TABS.index("Response")
    assert TABS.index("Model") < TABS.index("Validation") < TABS.index("Findings")
    assert TABS.index("Findings") < TABS.index("Next experiment")


def test_everything_is_locked_before_data_is_loaded():
    status = tab_status(SessionState())
    assert status["Data"]["unlocked"]
    for tab in TABS:
        if tab == "Data":
            continue
        assert not status[tab]["unlocked"], tab
        assert len(status[tab]["reason"]) > 20, tab


def test_response_is_locked_until_diagnostics_are_viewed():
    s = SessionState(data_loaded=True, diagnostics_viewed=False, analysis_id="attribution")
    assert tab_status(s)["Design diagnostics"]["unlocked"]
    assert not tab_status(s)["Response"]["unlocked"]
    s.diagnostics_viewed = True
    assert tab_status(s)["Response"]["unlocked"]


def test_model_is_locked_for_a_refused_response():
    s = SessionState(data_loaded=True, diagnostics_viewed=True, analysis_id="attribution",
                     response_chosen=True, response_usable=False)
    assert not tab_status(s)["Model"]["unlocked"]
    assert "refused" in tab_status(s)["Model"]["reason"].lower()


def test_findings_stay_locked_until_validation_passes():
    s = SessionState(data_loaded=True, diagnostics_viewed=True, response_chosen=True,
                     analysis_id="attribution", response_usable=True, model_configured=True,
                     validation_run=True, validation_passed=False)
    assert not tab_status(s)["Findings"]["unlocked"]
    assert "permutation" in tab_status(s)["Findings"]["reason"].lower()
    s.validation_passed = True
    assert tab_status(s)["Findings"]["unlocked"]


def test_next_experiment_is_open_even_when_nothing_was_supported():
    s = SessionState(data_loaded=True, diagnostics_viewed=True, response_chosen=True,
                     analysis_id="attribution", response_usable=True, model_configured=True,
                     validation_run=True, validation_passed=False)
    assert tab_status(s)["Next experiment"]["unlocked"]
    assert not tab_status(s)["Findings"]["unlocked"]


def test_gate_state_from_a_failed_run_locks_findings():
    ds = make_controlled("no_signal", seed=101)
    res = run_attribution(ds, "response_y", settings=FAST)
    s = state_from_result(res)
    assert s.validation_run and not s.validation_passed
    assert not tab_status(s)["Findings"]["unlocked"]


def test_gate_state_from_a_passing_run_unlocks_findings():
    ds = make_controlled("known_driver", seed=102)
    res = run_attribution(ds, "response_y", settings=FAST)
    s = state_from_result(res)
    assert s.validation_passed
    assert tab_status(s)["Findings"]["unlocked"]


# ---------------------------------------------------------------------
# the app module itself
# ---------------------------------------------------------------------
def test_app_source_compiles():
    import py_compile

    py_compile.compile(str(APP), doraise=True)


def test_app_uses_the_gate_and_the_banner():
    src = APP.read_text()
    assert "tab_status" in src and "st.stop()" in src
    assert "FINDINGS_BANNER" in src
    assert 'st.error(FINDINGS_BANNER)' in src          # banner is persistent, not optional


def test_app_never_calls_build_findings_directly():
    """Findings must come through build_report, which enforces the permutation gate."""
    src = APP.read_text()
    assert "build_findings" not in src


# ---------------------------------------------------------------------
# method record
# ---------------------------------------------------------------------
def test_method_record_has_every_required_part():
    ds = make_controlled("known_driver", seed=111)
    res = run_attribution(ds, "response_y", settings=FAST)
    md = to_markdown(res)
    for section in (
        "# Method record",
        "## Run",
        "## Design assessment",
        "## Validation",
        "## Decision rules applied",
        "## References",
    ):
        assert section in md, section
    assert "not a validated GMP output" in md
    assert "Annex 22" in md


def test_method_record_records_every_setting_and_version():
    ds = make_controlled("known_driver", seed=112)
    res = run_attribution(ds, "response_y", settings=FAST)
    rec = json.loads(to_json(res))
    assert rec["settings"]["seed"]
    assert rec["settings"]["n_permutations"] == FAST.n_permutations
    assert rec["libraries"]["scikit-learn"]
    assert rec["complexity_rule"]
    assert rec["validation"]["permutation_p"] is not None
    assert set(DECISION_RULES).issubset(rec["decision_rules"])
    assert len(rec["references"]) >= 10


def test_method_record_of_a_refusal_still_states_the_reason():
    ds = make_controlled("no_signal", seed=113)
    res = run_attribution(ds, "response_y", settings=FAST)
    rec = json.loads(to_json(res))
    assert rec["validation"]["verdict"] == "not_supported"
    assert "No attribution supported" in rec["headline"]
    assert rec["findings"] == []


def test_record_contains_no_causal_language():
    ds = make_controlled("known_driver", seed=114)
    res = run_attribution(ds, "response_y", settings=FAST)
    hits = causal_language_report(to_markdown(res))
    assert not hits, hits[:3]


def test_references_are_well_formed():
    for r in REFERENCES:
        assert set(r) == {"topic", "reference"}
        assert len(r["reference"]) > 30


def test_banner_text_is_the_one_the_prompt_asked_for():
    assert "Not causation" in FINDINGS_BANNER
    assert "confound list" in FINDINGS_BANNER
