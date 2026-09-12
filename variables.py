"""What can I analyse? The variable reference shown inside the app.

A formulator should not have to guess what to record. This lists every variable
the app understands, what family it belongs to, and why the family matters --
because the family decides whether something is offered as a knob to turn in the
confirmatory experiment or treated as a result that follows.

The list is a starting point, not a restriction: any numeric column you add will
be analysed. What the family tag changes is how it is treated.
"""
from __future__ import annotations

from typing import Dict, Iterable, List

import pandas as pd

FAMILY_MEANING: Dict[str, str] = {
    "MATERIAL": "What you received. Incoming attributes of API and excipients. You can "
    "select these (a grade, a supplier, a particle size band) but not dial them.",
    "FORMULATION": "What you designed. Quantities and levels in the recipe. Checked for "
    "the mixture constraint, because component percentages sum to a constant.",
    "PROCESS": "What you set on the equipment. These are the true knobs, and the ones the "
    "app proposes varying in a confirmatory experiment.",
    "IPQC_PHYSICAL": "What you measured on the way through. Real data, but a consequence "
    "of the settings above, so never proposed as a factor to vary - listed instead as "
    "'measured, expected to follow'.",
    "RESPONSE": "The result you want to explain. You attribute one of these at a time.",
}

# name, unit, note
_REF: Dict[str, List] = {
    "MATERIAL": [
        ("api_lot", "", "Lot identifier. Tag as META unless you convert it to a number."),
        ("api_d10_um", "um", "Fine end of the API distribution."),
        ("api_d50_um", "um", "Median API particle size. A common suspect for dissolution."),
        ("api_d90_um", "um", "Coarse end. Often what drives content uniformity risk."),
        ("api_span", "", "(D90 - D10) / D50. Width of the distribution."),
        ("api_ssa_m2_g", "m2/g", "Specific surface area."),
        ("api_bulk_density_g_ml", "g/mL", "Untapped."),
        ("api_tapped_density_g_ml", "g/mL", "After tapping."),
        ("api_moisture_pct", "% w/w", "Karl Fischer or LOD on the incoming API."),
        ("api_polymorph_form", "", "Categorical. Encode numerically or tag as META."),
        ("api_assay_pct", "%", "Potency of the incoming lot."),
        ("excipient_lot", "", "Lot identifier."),
        ("excipient_grade", "", "Categorical, e.g. HPMC K4M against K100M."),
        ("excipient_d50_um", "um", "Filler or polymer particle size."),
        ("excipient_moisture_pct", "% w/w", ""),
        ("excipient_viscosity_mpas", "mPa.s", "Polymer grade as a number, which is more "
         "useful than a grade name."),
        ("supplier", "", "Categorical."),
    ],
    "FORMULATION": [
        ("api_mg", "mg", "Quantity per unit."),
        ("api_pct_ww", "% w/w", ""),
        ("polymer_level_pct", "% w/w", "Release-controlling polymer, e.g. HPMC."),
        ("polymer_mg", "mg", "The same thing in mass. Do not model both: they are the same "
         "variable and the app will flag them as inseparable."),
        ("filler_pct_ww", "% w/w", "Lactose, MCC, mannitol."),
        ("binder_pct_ww", "% w/w", "PVP, HPC."),
        ("disintegrant_pct_ww", "% w/w", "Croscarmellose, sodium starch glycolate."),
        ("lubricant_pct_ww", "% w/w", "Magnesium stearate. Small changes matter."),
        ("glidant_pct_ww", "% w/w", "Talc, colloidal silica."),
        ("total_tablet_weight_mg", "mg", "Often constant, in which case it is excluded with "
         "that reason stated."),
        ("coating_pct_ww", "% w/w", "Target coating level."),
        ("drug_to_polymer_ratio", "", "A derived ratio; often more interpretable than either "
         "component alone."),
    ],
    "PROCESS": [
        ("granulation_water_pct", "% w/w", "Water or binder solution added."),
        ("granulation_water_addition_rate_g_min", "g/min", ""),
        ("granulation_torque_Nm", "N.m", "Endpoint by torque."),
        ("granulation_power_W", "W", "Endpoint by power draw."),
        ("granulation_amperage_A", "A", "Endpoint by current."),
        ("impeller_speed_rpm", "rpm", ""),
        ("chopper_speed_rpm", "rpm", ""),
        ("wet_massing_time_min", "min", ""),
        ("drying_inlet_temp_C", "degC", ""),
        ("drying_outlet_temp_C", "degC", ""),
        ("drying_time_min", "min", ""),
        ("drying_airflow_cmh", "m3/h", ""),
        ("milling_screen_mm", "mm", ""),
        ("milling_speed_rpm", "rpm", ""),
        ("blending_time_min", "min", ""),
        ("blender_speed_rpm", "rpm", ""),
        ("lubrication_time_min", "min", "Over-lubrication slows release; a frequent culprit."),
        ("main_compression_kN", "kN", ""),
        ("pre_compression_kN", "kN", ""),
        ("turret_speed_rpm", "rpm", "Sets dwell time."),
        ("feeder_speed_rpm", "rpm", ""),
        ("dwell_time_ms", "ms", "More mechanistic than turret speed if you can compute it."),
        ("coating_spray_rate_g_min", "g/min", ""),
        ("coating_inlet_temp_C", "degC", ""),
        ("coating_product_temp_C", "degC", ""),
        ("coating_pan_speed_rpm", "rpm", ""),
        ("coating_weight_gain_pct", "%", "Achieved, as opposed to the target above."),
        ("batch_size_kg", "kg", "Scale. Frequently confounded with everything else."),
    ],
    "IPQC_PHYSICAL": [
        ("blend_uniformity_rsd_pct", "%", ""),
        ("granule_d50_um", "um", ""),
        ("granule_lod_pct", "%", "Loss on drying after granulation."),
        ("bulk_density_g_ml", "g/mL", ""),
        ("tapped_density_g_ml", "g/mL", ""),
        ("carr_index", "", "Compressibility. Derived from the two densities."),
        ("hausner_ratio", "", "Also derived; do not expect it to separate from Carr index."),
        ("flow_time_s", "s", "Or angle of repose."),
        ("weight_mg", "mg", "Mean tablet weight."),
        ("weight_rsd_pct", "%", ""),
        ("thickness_mm", "mm", ""),
        ("hardness_N", "N", "A result of compression force, not a setting."),
        ("friability_pct", "%", ""),
        ("disintegration_min", "min", "A response in its own right, or a predictor of "
         "dissolution."),
        ("lod_pct", "%", "Final blend or tablet."),
    ],
    "RESPONSE": [
        ("assay_pct", "% LC", "Percent of label claim."),
        ("content_uniformity_av", "", "Acceptance value, USP <905>."),
        ("content_uniformity_rsd_pct", "%", ""),
        ("related_substances_total_pct", "%", ""),
        ("related_substances_max_individual_pct", "%", ""),
        ("q30_pct", "%", "Released at 30 minutes. Any single time point works."),
        ("q45_pct", "%", ""),
        ("q12h_pct", "%", "For extended release."),
        ("dissolution_mdt_h", "h", "Computed for you from a profile table."),
        ("water_content_pct", "%", ""),
        ("hardness_N", "N", "Can be the response instead of a predictor."),
        ("disintegration_min", "min", "Likewise."),
    ],
}

PROFILE_DERIVED = [
    ("weibull_td", "h", "Time to about 63% released. The scale of the curve."),
    ("weibull_beta", "", "Shape. Below 1 is a steep early release."),
    ("first_order_k", "1/h", "First-order rate constant."),
    ("mdt", "h", "Mean dissolution time, model independent."),
    ("dissolution_efficiency", "%", "Area under the curve as a percentage of complete "
     "release over the same window."),
    ("t50", "h", "Time to 50% released. Blank if never reached; not extrapolated."),
    ("t80", "h", "Time to 80% released."),
    ("f2_vs_reference", "", "Similarity to a reference batch you nominate."),
    ("cross_medium_mdt_spread", "h", "Spread of MDT across media. Often the most "
     "diagnostic response of all, and frequently overlooked."),
]


def reference_frame(families: Iterable[str] | None = None) -> pd.DataFrame:
    """The whole reference as a table, for display and filtering."""
    rows = []
    for fam, entries in _REF.items():
        if families and fam not in set(families):
            continue
        for name, unit, note in entries:
            rows.append({"family": fam, "variable": name, "unit": unit, "note": note})
    return pd.DataFrame(rows)


def profile_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [{"metric": n, "unit": u, "note": d} for n, u, d in PROFILE_DERIVED]
    )


def families_for(names: Iterable[str]) -> Dict[str, str]:
    """Map chosen reference variables back to their families.

    A few variables appear in two families on purpose: hardness and disintegration
    time are IPQC results that can also be the thing you are trying to explain. The
    first listing wins here, so they arrive tagged IPQC_PHYSICAL; retag them RESPONSE
    on the Data tab if that is the question you are asking.
    """
    lookup: Dict[str, str] = {}
    for fam, entries in _REF.items():
        for name, _, _ in entries:
            lookup.setdefault(name, fam)      # first family listed wins
    return {n: lookup.get(n, "PROCESS") for n in names}


DUAL_ROLE = sorted(
    {
        name
        for fam, entries in _REF.items()
        for name, _, _ in entries
        if sum(1 for f2, e2 in _REF.items() for n2, _, _ in e2 if n2 == name) > 1
    }
)


def build_template(
    names: Iterable[str], n_batches: int = 6, include_dates: bool = True
) -> pd.DataFrame:
    """A blank batch table with the variables you picked, ready to fill in."""
    cols = ["batch", "formulation"]
    if include_dates:
        cols.append("manufacturing_date")
    chosen = [n for n in names if n not in cols]
    df = pd.DataFrame({c: [None] * n_batches for c in cols + chosen})
    df["batch"] = [f"B{i+1:03d}" for i in range(n_batches)]
    return df


def dissolution_template(
    batches: Iterable[str] | None = None,
    media: Iterable[str] = ("0.1N HCl",),
    times=(0.25, 0.5, 1, 2, 4, 6, 8, 12),
) -> pd.DataFrame:
    rows = []
    for b in (batches or ["B001", "B002"]):
        for m in media:
            for t in times:
                rows.append({"batch": b, "medium": m, "time_h": t, "percent_released": None})
    return pd.DataFrame(rows)


def count_variables() -> Dict[str, int]:
    return {fam: len(entries) for fam, entries in _REF.items()}
