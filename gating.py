"""Prompt 7 -- tab gating, kept out of the interface so it can be tested.

The tab order is the order the analysis must be done in. Users will want to skip
to the answer; these rules are what stops them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

TABS = (
    "Data",
    "Design diagnostics",
    "Response",
    "Model",
    "Validation",
    "Findings",
    "Next experiment",
    "Method record",
)

FINDINGS_BANNER = (
    "Association from observational batch data. Not causation. "
    "See the confound list for each factor."
)


@dataclass
class SessionState:
    """What has actually been done, not what the user would like to have been done."""

    data_loaded: bool = False
    data_has_errors: bool = False
    diagnostics_viewed: bool = False
    response_chosen: bool = False
    response_usable: bool = False
    model_configured: bool = False
    validation_run: bool = False
    validation_passed: bool = False
    refused: bool = False
    notes: Dict[str, str] = field(default_factory=dict)


def tab_status(state: SessionState) -> Dict[str, Dict]:
    """For each tab: unlocked or not, and the reason when it is not."""
    s = state
    out: Dict[str, Dict] = {}

    def put(name, ok, reason=""):
        out[name] = {"unlocked": bool(ok), "reason": reason}

    put("Data", True)
    put(
        "Design diagnostics",
        s.data_loaded,
        "Load a batch table and its metadata first.",
    )
    put(
        "Response",
        s.data_loaded and s.diagnostics_viewed,
        "Read the design diagnostics first. What the data can answer is decided there, "
        "not by the response you pick.",
    )
    put(
        "Model",
        s.response_chosen and s.response_usable,
        "Choose a response that passed its own checks. A response the app has refused "
        "cannot be modelled by choosing a different model.",
    )
    put(
        "Validation",
        s.model_configured,
        "Configure and fit a model first.",
    )
    put(
        "Findings",
        s.validation_run and s.validation_passed and not s.refused,
        "Validation has not been run, or the model did not beat its permutation null. "
        "There are no findings to show: an importance ranking from a model that failed "
        "its null is a picture of noise.",
    )
    put(
        "Next experiment",
        s.validation_run,
        "Run validation first. The proposed experiment is built from what the data "
        "could not settle, so it needs the validation result either way.",
    )
    put(
        "Method record",
        s.data_loaded,
        "Load data first. The record documents an actual run: settings, exclusions, "
        "decision rules and versions.",
    )
    return out


def unlocked_tabs(state: SessionState) -> List[str]:
    st = tab_status(state)
    return [t for t in TABS if st[t]["unlocked"]]


def state_from_result(result, diagnostics_viewed: bool = True) -> SessionState:
    """Derive the gate state from an AnalysisResult."""
    if result is None:
        return SessionState()
    return SessionState(
        data_loaded=True,
        diagnostics_viewed=diagnostics_viewed,
        response_chosen=True,
        response_usable=bool(result.response.usable),
        model_configured=result.model is not None or result.validation is not None,
        validation_run=result.validation is not None,
        validation_passed=bool(result.validation is not None and result.validation.passed),
        refused=bool(result.refused),
    )
