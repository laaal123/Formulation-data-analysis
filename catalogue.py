"""What can I actually do here?

Every analysis the app offers, named the way a formulator would ask for it, with
the honest minimum for each. Some need two batches. Some need twenty. The point of
listing them together is that the small-batch ones are real analyses, not consolation
prizes: with three batches, "what differs between these two" is exactly the right
question and "which factor drives dissolution" is exactly the wrong one.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from loader import BatchDataset

DESCRIPTIVE = "Describe and compare"
PROFILE = "Dissolution profiles"
ATTRIBUTION = "Find what is linked to a result"


@dataclass
class Analysis:
    id: str
    name: str                       # what a formulator would call it
    question: str                   # the question it answers, in one line
    group: str
    min_batches: int
    min_formulations: int = 1
    needs_dissolution: bool = False
    needs_response: bool = False
    output: str = ""
    caution: str = ""

    def availability(self, ds: Optional[BatchDataset]) -> Dict:
        if ds is None:
            return {"available": False, "reason": "Load or enter data first."}
        reasons = []
        if ds.n_batches < self.min_batches:
            reasons.append(
                f"needs at least {self.min_batches} batches, you have {ds.n_batches}"
            )
        if ds.n_formulations < self.min_formulations:
            reasons.append(
                f"needs at least {self.min_formulations} formulations, you have "
                f"{ds.n_formulations}"
            )
        if self.needs_dissolution and ds.dissolution is None:
            reasons.append("needs a dissolution profile table")
        if self.needs_response and not ds.response_columns():
            reasons.append("needs at least one column tagged RESPONSE")
        return {"available": not reasons, "reason": "; ".join(reasons)}


CATALOGUE: List[Analysis] = [
    # ---------------- 2 batches upward -----------------------------
    Analysis(
        id="compare",
        name="Compare batches side by side",
        question="How do these batches differ, variable by variable?",
        group=DESCRIPTIVE,
        min_batches=2,
        output="One column per batch, one row per variable, grouped by family.",
    ),
    Analysis(
        id="what_changed",
        name="What changed between two batches?",
        question="My last batch behaved differently. What was not the same?",
        group=DESCRIPTIVE,
        min_batches=2,
        output="Every variable that differs, largest difference first, with the size of "
        "the gap in units, in percent, and against the spread of your other batches.",
        caution="A difference list, never a cause. With two batches every difference is "
        "perfectly confounded with every other, so this tells you what to test next.",
    ),
    Analysis(
        id="descriptive",
        name="Summary statistics for every variable",
        question="What are the means, spreads and ranges across my batches?",
        group=DESCRIPTIVE,
        min_batches=2,
        output="n, mean, SD, RSD, min, max and range per variable.",
    ),
    Analysis(
        id="trend",
        name="Trend one variable across batches",
        question="Is this value drifting over time or between formulations?",
        group=DESCRIPTIVE,
        min_batches=3,
        output="The variable in manufacturing order with its descriptive mean and spread.",
        caution="Not a control chart. Control limits need a stable process and many more "
        "batches than a formulation series has.",
    ),
    Analysis(
        id="spec_check",
        name="Check batches against your specification",
        question="Which batches sit outside the limits I set?",
        group=DESCRIPTIVE,
        min_batches=1,
        output="Pass or fail per batch per variable against limits you enter.",
        caution="A comparison against limits you typed. Not a release decision.",
    ),
    # ---------------- dissolution ----------------------------------
    Analysis(
        id="f2",
        name="Compare two dissolution profiles (f2 and f1)",
        question="Is my test batch similar to the reference?",
        group=PROFILE,
        min_batches=2,
        needs_dissolution=True,
        output="f2 similarity, f1 difference, the point-by-point table, and the verdict "
        "against the conventional f2 >= 50 limit.",
        caution="Flags the case where too many points sit above 85% released, which "
        "inflates f2.",
    ),
    Analysis(
        id="profile_summary",
        name="Summarise every dissolution curve",
        question="What are the release characteristics of each batch as numbers?",
        group=PROFILE,
        min_batches=1,
        needs_dissolution=True,
        output="Weibull scale and shape, mean dissolution time, dissolution efficiency, "
        "t50 and t80 per batch and medium.",
    ),
    # ---------------- attribution ----------------------------------
    Analysis(
        id="design_check",
        name="What can my data actually answer?",
        question="Before I model anything: which questions are answerable with what I have?",
        group=ATTRIBUTION,
        min_batches=3,
        output="Factors that move together and cannot be separated, factors that are just "
        "a formulation label in disguise, mixture constraints, and a plain statement of "
        "what is and is not answerable.",
        caution="Run this first. It often ends the investigation by showing the answer "
        "is not in the data.",
    ),
    Analysis(
        id="screen",
        name="Screen all factors against one result",
        question="Which factors track my CQA, one at a time?",
        group=ATTRIBUTION,
        min_batches=8,
        needs_response=True,
        output="Correlation of each factor with the chosen response, with false discovery "
        "rate control.",
        caution="A screen, not a finding. It ignores every other factor and all the "
        "confounding between them.",
    ),
    Analysis(
        id="attribution",
        name="Which factors are linked to my result? (full attribution)",
        question="What is associated with my dissolution, assay or hardness, and can I "
        "trust it?",
        group=ATTRIBUTION,
        min_batches=10,
        min_formulations=2,
        needs_response=True,
        output="A ranked list of associated factors with effect sizes, confidence "
        "intervals, stability, and the confounders attached to each; validated by "
        "held-out formulations against a shuffled-response null.",
        caution="Refuses when the data cannot support a conclusion, which is common below "
        "about 20 batches.",
    ),
    Analysis(
        id="next_experiment",
        name="Design the experiment that would settle it",
        question="What should I actually run to find out?",
        group=ATTRIBUTION,
        min_batches=3,
        output="The smallest factorial that separates the leading candidate from the "
        "factors it currently travels with: which factors, what ranges, what to hold "
        "constant, how many batches.",
    ),
]

BY_ID: Dict[str, Analysis] = {a.id: a for a in CATALOGUE}
GROUPS: List[str] = [DESCRIPTIVE, PROFILE, ATTRIBUTION]


def available_analyses(ds: Optional[BatchDataset]) -> List[Dict]:
    out = []
    for a in CATALOGUE:
        av = a.availability(ds)
        out.append({"analysis": a, **av})
    return out


def small_batch_guidance(n_batches: int, n_formulations: int) -> str:
    """What is honestly possible at this size."""
    if n_batches < 2:
        return (
            "One batch. You can check it against a specification and summarise its "
            "dissolution curve. Nothing can be compared and nothing can be attributed."
        )
    if n_batches < 5:
        return (
            f"{n_batches} batches. Comparison, profile similarity and a difference list "
            "between any two batches are all valid and useful here. Factor attribution is "
            "not: with this many batches and dozens of recorded variables, every "
            "difference is perfectly confounded with every other, and any ranking of "
            "causes would be an artefact of the arithmetic rather than a property of your "
            "product. The difference list is the right tool at this size, and it feeds "
            "directly into a designed experiment."
        )
    if n_batches < 10:
        return (
            f"{n_batches} batches. Comparison, trending and profile work are solid. The "
            "design check will run and is worth reading. Full attribution is still out of "
            "reach and will usually be refused after validation."
        )
    if n_formulations < 2:
        return (
            f"{n_batches} batches but only one formulation. Anything that changed between "
            "formulations cannot be assessed, because there is nothing to hold out. "
            "Factors that vary batch to batch within this formulation can be looked at."
        )
    if n_batches < 20:
        return (
            f"{n_batches} batches across {n_formulations} formulations. Attribution will "
            "run. Expect it to refuse fairly often; that refusal is information, not a "
            "failure of the tool."
        )
    return (
        f"{n_batches} batches across {n_formulations} formulations. All analyses are "
        "available. The design check still decides what is answerable."
    )
