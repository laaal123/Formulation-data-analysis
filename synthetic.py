"""Synthetic example data with known ground truth.

Deliberately built the way a real formulation series is built: factors were
changed together between formulations, so the true driver is not separable from
its travelling companions. The app is correct only if it says so.
"""
from __future__ import annotations

from typing import Dict, Tuple

import numpy as np
import pandas as pd

from config import RANDOM_SEED
from loader import BatchDataset, build_dataset

# ground truth of the example dataset
GROUND_TRUTH = {
    "causal_driver": "hpmc_level_pct",
    "inseparable_cluster": ["hpmc_level_pct", "main_compression_kN", "hardness_N"],
    "secondary_driver": "granulation_water_pct",
    "noise_predictors_prefix": "noise_",
    "response_formula": (
        "Weibull td increases with hpmc_level_pct and with granulation_water_pct; "
        "Q30 and MDT follow from the profile."
    ),
}

MEDIA = ("0.1N HCl", "pH 4.5 acetate", "pH 6.8 phosphate")
TIME_POINTS_H = (0.25, 0.5, 1.0, 2.0, 4.0, 6.0, 8.0, 12.0)


def _weibull_release(t: np.ndarray, td: float, beta: float) -> np.ndarray:
    return 100.0 * (1.0 - np.exp(-((t / td) ** beta)))


def make_synthetic(
    n_formulations: int = 4,
    batches_per_formulation: int = 6,
    n_noise: int = 40,
    seed: int = RANDOM_SEED,
    signal: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, str], pd.DataFrame, pd.DataFrame]:
    """Return (wide, families, metadata, dissolution)."""
    rng = np.random.default_rng(seed)
    n = n_formulations * batches_per_formulation

    formulation = np.repeat([f"F{i+1}" for i in range(n_formulations)], batches_per_formulation)
    batch = [f"B{i+1:03d}" for i in range(n)]

    # --- factors that were changed together between formulations -----
    hpmc_by_f = np.linspace(18.0, 24.0, n_formulations)
    force_by_f = np.linspace(8.0, 16.0, n_formulations)
    idx = np.repeat(np.arange(n_formulations), batches_per_formulation)

    hpmc = hpmc_by_f[idx] + rng.normal(0, 0.35, n)
    force = force_by_f[idx] + rng.normal(0, 0.55, n)          # r ~ 0.95 with hpmc
    hardness = 4.0 * force + rng.normal(0, 3.0, n) + 20.0      # rides on force

    # --- factors that vary within formulation (genuinely estimable) ---
    water = rng.normal(28.0, 2.2, n)                           # granulation water % w/w
    torque = 0.55 * water + rng.normal(0, 0.5, n) + 4.0
    blend_time = rng.choice([5.0, 10.0, 15.0], n)
    lub_time = rng.choice([2.0, 3.0, 4.0], n)
    lod = rng.normal(1.8, 0.25, n)
    turret = rng.normal(30.0, 3.0, n)

    # --- mixture: percentages that sum to 100 -------------------------
    api_pct = np.full(n, 25.0)
    mg_st_pct = np.full(n, 1.0) + rng.normal(0, 0.05, n)
    talc_pct = np.full(n, 1.5) + rng.normal(0, 0.05, n)
    mcc_pct = np.full(n, 15.0) + rng.normal(0, 0.4, n)
    lactose_pct = 100.0 - (api_pct + hpmc + mg_st_pct + talc_pct + mcc_pct)

    total_weight = 500.0  # constant column, deliberately
    api_mg = api_pct / 100.0 * total_weight
    hpmc_mg = hpmc / 100.0 * total_weight

    # --- response generated from the true model -----------------------
    if signal:
        td = 1.6 + 0.42 * (hpmc - 18.0) + 0.05 * (water - 28.0)
    else:
        td = np.full(n, 3.0)
    td = np.clip(td + rng.normal(0, 0.12, n), 0.4, None)
    beta = 0.95 + rng.normal(0, 0.03, n)

    diss_rows = []
    medium_factor = {"0.1N HCl": 1.0, "pH 4.5 acetate": 1.12, "pH 6.8 phosphate": 1.25}
    t = np.array(TIME_POINTS_H)
    for i, b in enumerate(batch):
        for medium, mf in medium_factor.items():
            rel = _weibull_release(t, td[i] * mf, beta[i])
            rel = np.clip(rel + rng.normal(0, 1.6, t.size), 0, 105)
            rel = np.maximum.accumulate(rel)  # monotone, as a real profile is
            for tt, rr in zip(t, rel):
                diss_rows.append(
                    {
                        "batch": b,
                        "medium": medium,
                        "time_h": float(tt),
                        "percent_released": float(rr),
                    }
                )
    dissolution = pd.DataFrame(diss_rows)

    q30 = (
        dissolution.query("medium == '0.1N HCl' and time_h == 0.5")
        .set_index("batch")["percent_released"]
        .reindex(batch)
        .to_numpy()
    )

    assay = 99.0 + rng.normal(0, 1.1, n)
    av = np.abs(rng.normal(4.0, 1.6, n))
    rs_total = np.abs(rng.normal(0.25, 0.06, n))

    data = {
        "batch": batch,
        # MATERIAL
        "api_d10_um": rng.normal(8, 1.2, n),
        "api_d50_um": rng.normal(35, 5, n),
        "api_d90_um": rng.normal(95, 12, n),
        "api_bulk_density_g_ml": rng.normal(0.42, 0.03, n),
        "api_tapped_density_g_ml": rng.normal(0.58, 0.03, n),
        "api_moisture_pct": rng.normal(0.35, 0.06, n),
        "excipient_d50_um": rng.normal(120, 15, n),
        # FORMULATION
        "api_pct_ww": api_pct,
        "hpmc_level_pct": hpmc,
        "mcc_pct_ww": mcc_pct,
        "lactose_pct_ww": lactose_pct,
        "mg_stearate_pct_ww": mg_st_pct,
        "talc_pct_ww": talc_pct,
        "api_mg": api_mg,
        "hpmc_mg": hpmc_mg,
        "total_tablet_weight_mg": np.full(n, total_weight),  # constant on purpose
        # PROCESS
        "granulation_water_pct": water,
        "granulation_torque_Nm": torque,
        "impeller_speed_rpm": rng.normal(300, 25, n),
        "drying_inlet_temp_C": rng.normal(60, 3, n),
        "drying_outlet_temp_C": rng.normal(38, 2, n),
        "milling_screen_mm": rng.choice([1.0, 1.5], n),
        "blending_time_min": blend_time,
        "lubrication_time_min": lub_time,
        "main_compression_kN": force,
        "pre_compression_kN": 0.2 * force + rng.normal(0, 0.15, n),
        "turret_speed_rpm": turret,
        "feeder_speed_rpm": rng.normal(25, 2.5, n),
        # IPQC_PHYSICAL
        "hardness_N": hardness,
        "thickness_mm": rng.normal(5.2, 0.06, n),
        "weight_mg": rng.normal(total_weight, 4.0, n),
        "friability_pct": np.abs(rng.normal(0.25, 0.07, n)),
        "disintegration_min": rng.normal(9, 1.5, n),
        "lod_pct": lod,
        "bulk_density_g_ml": rng.normal(0.46, 0.02, n),
        "tapped_density_g_ml": rng.normal(0.61, 0.02, n),
        # RESPONSE
        "assay_pct": assay,
        "content_uniformity_av": av,
        "related_substances_total_pct": rs_total,
        "q30_pct": q30,
    }

    # a column with heavy missingness, on purpose
    partial = rng.normal(0.9, 0.1, n)
    mask = rng.random(n) < 0.5
    partial[mask] = np.nan
    data["coating_weight_gain_pct"] = partial

    for j in range(n_noise):
        data[f"noise_{j:02d}"] = rng.normal(0, 1, n)

    wide = pd.DataFrame(data)

    families: Dict[str, str] = {}
    material = [
        "api_d10_um", "api_d50_um", "api_d90_um", "api_bulk_density_g_ml",
        "api_tapped_density_g_ml", "api_moisture_pct", "excipient_d50_um",
    ]
    formulation_cols = [
        "api_pct_ww", "hpmc_level_pct", "mcc_pct_ww", "lactose_pct_ww",
        "mg_stearate_pct_ww", "talc_pct_ww", "api_mg", "hpmc_mg",
        "total_tablet_weight_mg",
    ]
    process = [
        "granulation_water_pct", "granulation_torque_Nm", "impeller_speed_rpm",
        "drying_inlet_temp_C", "drying_outlet_temp_C", "milling_screen_mm",
        "blending_time_min", "lubrication_time_min", "main_compression_kN",
        "pre_compression_kN", "turret_speed_rpm", "feeder_speed_rpm",
        "coating_weight_gain_pct",
    ]
    ipqc = [
        "hardness_N", "thickness_mm", "weight_mg", "friability_pct",
        "disintegration_min", "lod_pct", "bulk_density_g_ml", "tapped_density_g_ml",
    ]
    responses = ["assay_pct", "content_uniformity_av", "related_substances_total_pct", "q30_pct"]
    for c in material:
        families[c] = "MATERIAL"
    for c in formulation_cols:
        families[c] = "FORMULATION"
    for c in process:
        families[c] = "PROCESS"
    for c in ipqc:
        families[c] = "IPQC_PHYSICAL"
    for c in responses:
        families[c] = "RESPONSE"
    for j in range(n_noise):
        families[f"noise_{j:02d}"] = "PROCESS"

    start = pd.Timestamp("2023-01-10")
    dates = [start + pd.Timedelta(days=int(14 * i + rng.integers(0, 4))) for i in range(n)]
    metadata = pd.DataFrame(
        {
            "batch": batch,
            "formulation": formulation,
            "manufacturing_date": dates,
            "scale": np.where(idx < 2, "pilot", "pilot"),
            "equipment_train": "Train A",
            "site": "Site 1",
            "operator_shift": rng.choice(["day", "night"], n),
        }
    )
    return wide, families, metadata, dissolution


def make_controlled(
    mode: str = "known_driver",
    n_formulations: int = 6,
    batches_per_formulation: int = 5,
    n_noise: int = 40,
    seed: int = RANDOM_SEED,
    effect: float = 3.0,
    noise_sd: float = 1.0,
) -> BatchDataset:
    """Small, controlled datasets with ground truth, for the correctness tests.

    mode:
      known_driver : y = effect * driver + noise, driver varies WITHIN formulation
      no_signal    : y is pure noise, unrelated to every predictor
      confounded   : x1 and x2 correlated at ~0.97, only x1 enters y
    """
    rng = np.random.default_rng(seed)
    n = n_formulations * batches_per_formulation
    idx = np.repeat(np.arange(n_formulations), batches_per_formulation)
    batch = [f"C{i+1:03d}" for i in range(n)]
    formulation = np.array([f"F{i+1}" for i in idx])

    data = {"batch": batch}
    families: Dict[str, str] = {}

    driver = rng.normal(0, 1, n)
    data["driver"] = driver
    families["driver"] = "PROCESS"

    if mode == "confounded":
        partner = 0.97 * driver + np.sqrt(1 - 0.97**2) * rng.normal(0, 1, n)
        data["partner"] = partner
        families["partner"] = "PROCESS"

    for j in range(n_noise):
        data[f"noise_{j:02d}"] = rng.normal(0, 1, n)
        families[f"noise_{j:02d}"] = "PROCESS"

    if mode == "no_signal":
        y = rng.normal(50, noise_sd, n)
    else:
        y = 50.0 + effect * driver + rng.normal(0, noise_sd, n)
    data["response_y"] = y
    families["response_y"] = "RESPONSE"

    wide = pd.DataFrame(data)
    start = pd.Timestamp("2024-01-05")
    metadata = pd.DataFrame(
        {
            "batch": batch,
            "formulation": formulation,
            "manufacturing_date": [start + pd.Timedelta(days=int(7 * i)) for i in range(n)],
            "scale": "pilot",
            "equipment_train": "Train A",
            "site": "Site 1",
            "operator_shift": "day",
        }
    )
    return build_dataset(wide, families, metadata)


def make_mixture(
    n_formulations: int = 6,
    batches_per_formulation: int = 5,
    n_noise: int = 10,
    seed: int = RANDOM_SEED,
    betas: Tuple[float, ...] = (40.0, 10.0, 25.0, 5.0),
    noise_sd: float = 1.5,
) -> BatchDataset:
    """A genuine mixture with known component coefficients.

    Four components sum to 100% w/w and the response is a linear Scheffe model in
    them. Components vary within formulations, so leave-one-formulation-out is a fair
    test rather than extrapolation.
    """
    rng = np.random.default_rng(seed)
    n = n_formulations * batches_per_formulation
    idx = np.repeat(np.arange(n_formulations), batches_per_formulation)
    batch = [f"M{i+1:03d}" for i in range(n)]
    names = ["comp_a_pct_ww", "comp_b_pct_ww", "comp_c_pct_ww", "comp_d_pct_ww"]

    base = np.array([0.40, 0.25, 0.25, 0.10])
    P = np.empty((n, 4))
    for i in range(n):
        shift = np.zeros(4)
        shift[idx[i] % 4] = 0.05                      # formulations sit in different regions
        P[i] = rng.dirichlet((base + shift) * 120.0)  # within-formulation spread as well

    y = P @ np.array(betas) * 100.0 + rng.normal(0, noise_sd, n)

    data = {"batch": batch}
    families: Dict[str, str] = {}
    for j, nm in enumerate(names):
        data[nm] = P[:, j] * 100.0
        families[nm] = "FORMULATION"
    for j in range(n_noise):
        data[f"noise_{j:02d}"] = rng.normal(0, 1, n)
        families[f"noise_{j:02d}"] = "PROCESS"
    data["response_y"] = y
    families["response_y"] = "RESPONSE"

    metadata = pd.DataFrame(
        {
            "batch": batch,
            "formulation": [f"F{i+1}" for i in idx],
            "manufacturing_date": [
                pd.Timestamp("2024-02-01") + pd.Timedelta(days=int(5 * i)) for i in range(n)
            ],
            "scale": "pilot",
            "equipment_train": "Train A",
            "site": "Site 1",
            "operator_shift": "day",
        }
    )
    ds = build_dataset(pd.DataFrame(data), families, metadata)
    ds.ground_truth_betas = dict(zip(names, betas))
    return ds


def example_dataset(seed: int = RANDOM_SEED, signal: bool = True, **kwargs) -> BatchDataset:
    wide, families, metadata, dissolution = make_synthetic(seed=seed, signal=signal, **kwargs)
    return build_dataset(wide, families, metadata, dissolution)
