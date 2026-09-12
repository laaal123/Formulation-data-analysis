"""Global configuration. Everything that could introduce non-determinism lives here.

Annex 22 draft expects static, deterministic models with logged settings.
One seed, one place, logged into every result object.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Dict

RANDOM_SEED = 20240101

# Column families. The family decides how a column is treated downstream.
FAMILIES = (
    "MATERIAL",
    "FORMULATION",
    "PROCESS",
    "IPQC_PHYSICAL",
    "RESPONSE",
    "META",
)

# Families that may be offered as candidate predictors.
PREDICTOR_FAMILIES = ("MATERIAL", "FORMULATION", "PROCESS", "IPQC_PHYSICAL")

# Thresholds, all named so they can be shown to the user and cited in the record.
THRESHOLDS = {
    "missing_fraction_flag": 0.40,      # prompt 1: report columns >40% missing
    "collinear_cluster_r": 0.90,        # prompt 2: |r| > 0.9 -> cannot separate
    "vif_flag": 10.0,
    "formulation_alias_r2": 0.95,       # prompt 2: predictor is a formulation label
    "time_alias_r2": 0.95,
    "mixture_sum_tolerance": 0.5,       # % w/w tolerance for "sums to a constant"
    "p_over_n_flag": 1 / 3,             # prompt 2: flag when p > n/3
    "stability_unstable": 0.60,         # prompt 5: below 60% is unstable
    "permutation_alpha": 0.05,
    "min_n_for_nonlinear": 30,          # prompt 4
    "max_depth_small_n": 3,             # prompt 4
}


@dataclass
class RunSettings:
    """Every setting that affects a result. Copied into result objects verbatim."""

    seed: int = RANDOM_SEED
    n_permutations: int = 1000
    n_bootstrap: int = 500
    inner_cv_splits: int = 5
    collinear_cluster_r: float = THRESHOLDS["collinear_cluster_r"]
    stability_threshold: float = THRESHOLDS["stability_unstable"]
    permutation_alpha: float = THRESHOLDS["permutation_alpha"]
    extra: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def library_versions() -> Dict[str, str]:
    """Pinned-version evidence for the method record."""
    import numpy
    import pandas
    import scipy
    import sklearn
    import sys

    return {
        "python": sys.version.split()[0],
        "numpy": numpy.__version__,
        "pandas": pandas.__version__,
        "scipy": scipy.__version__,
        "scikit-learn": sklearn.__version__,
    }
