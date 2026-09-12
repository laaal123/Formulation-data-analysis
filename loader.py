"""Prompt 1 -- data layer.

Wide or long batch table, batch metadata, dissolution profiles.
Validation reports; nothing is silently coerced and nothing is silently imputed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

from config import FAMILIES, PREDICTOR_FAMILIES, THRESHOLDS
from errors import DataError

SEVERITIES = ("ERROR", "WARNING", "INFO")


@dataclass
class Issue:
    severity: str
    code: str
    message: str
    columns: List[str] = field(default_factory=list)
    detail: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if self.severity not in SEVERITIES:
            raise ValueError(f"severity must be one of {SEVERITIES}")


@dataclass
class ValidationReport:
    issues: List[Issue] = field(default_factory=list)

    def add(self, severity: str, code: str, message: str, columns=None, **detail) -> None:
        self.issues.append(
            Issue(severity, code, message, list(columns or []), dict(detail))
        )

    # -- queries -------------------------------------------------------
    @property
    def errors(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "ERROR"]

    @property
    def warnings(self) -> List[Issue]:
        return [i for i in self.issues if i.severity == "WARNING"]

    @property
    def ok(self) -> bool:
        return not self.errors

    def codes(self) -> List[str]:
        return [i.code for i in self.issues]

    def has(self, code: str) -> bool:
        return code in self.codes()

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            [
                {
                    "severity": i.severity,
                    "code": i.code,
                    "message": i.message,
                    "columns": ", ".join(i.columns),
                }
                for i in self.issues
            ],
            columns=["severity", "code", "message", "columns"],
        )

    def text(self) -> str:
        if not self.issues:
            return "No validation issues found."
        order = {"ERROR": 0, "WARNING": 1, "INFO": 2}
        lines = []
        for i in sorted(self.issues, key=lambda x: order[x.severity]):
            cols = f"  [{', '.join(i.columns)}]" if i.columns else ""
            lines.append(f"{i.severity}: {i.message}{cols}")
        return "\n".join(lines)


# ---------------------------------------------------------------------
# Ingestion helpers
# ---------------------------------------------------------------------
def read_table(path: str | Path) -> pd.DataFrame:
    """Read a CSV or Excel file. Nothing is coerced here."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix in (".csv", ".txt"):
        return pd.read_csv(path)
    if suffix in (".xlsx", ".xlsm", ".xls"):
        return pd.read_excel(path)
    raise ValueError(f"Unsupported file type: {suffix}")


def long_to_wide(
    long_df: pd.DataFrame,
    batch_col: str = "batch",
    variable_col: str = "variable",
    value_col: str = "value",
) -> pd.DataFrame:
    """Pivot a (batch, variable, value) table to one row per batch.

    Duplicate (batch, variable) pairs are an error, not something to average away.
    """
    missing = [c for c in (batch_col, variable_col, value_col) if c not in long_df.columns]
    if missing:
        raise ValueError(f"Long table missing columns: {missing}")
    dupes = long_df.duplicated(subset=[batch_col, variable_col])
    if dupes.any():
        pairs = long_df.loc[dupes, [batch_col, variable_col]].astype(str).agg(" / ".join, axis=1)
        raise ValueError(
            "Duplicate (batch, variable) pairs in long table; refusing to aggregate: "
            + ", ".join(sorted(set(pairs))[:10])
        )
    wide = long_df.pivot(index=batch_col, columns=variable_col, values=value_col)
    wide.columns.name = None
    return wide.reset_index()


META_COLUMNS = (
    "formulation",
    "manufacturing_date",
    "scale",
    "equipment_train",
    "site",
    "operator_shift",
)


def split_single_table(df: pd.DataFrame, batch_col: str = "batch"):
    """Split one combined table into (wide, metadata).

    Typing or pasting two separate tables is a nuisance, so the app accepts a single
    table that carries `formulation` alongside the measurements and separates them
    here. Anything recognised as metadata is moved out; everything else stays as a
    candidate variable.
    """
    if batch_col not in df.columns:
        raise DataError(
            f"The table needs a column named {batch_col!r}. Found: "
            + ", ".join(map(str, df.columns[:12]))
        )
    if "formulation" not in df.columns:
        raise DataError(
            "The table needs a 'formulation' column saying which formulation each batch "
            "belongs to. Without it there is no grouping to validate against, and "
            "leave-one-formulation-out cannot be done."
        )
    present = [c for c in META_COLUMNS if c in df.columns]
    metadata = df[[batch_col] + present].copy()
    wide = df.drop(columns=present)
    return wide, metadata


def blank_entry_table(n_batches: int = 6, extra_columns=None) -> pd.DataFrame:
    """An empty grid for manual entry, with the two required columns already present."""
    cols = ["batch", "formulation"] + list(extra_columns or [])
    data = {c: [None] * n_batches for c in cols}
    df = pd.DataFrame(data)
    df["batch"] = [f"B{i+1:03d}" for i in range(n_batches)]
    return df


# ---------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------
@dataclass
class BatchDataset:
    """One row per batch, plus metadata, plus optional dissolution profiles."""

    wide: pd.DataFrame                      # indexed by batch id
    families: Dict[str, str]                # column -> family
    metadata: pd.DataFrame                  # indexed by batch id
    dissolution: Optional[pd.DataFrame] = None   # long: batch, medium, time_h, percent_released
    report: ValidationReport = field(default_factory=ValidationReport)
    imputation_record: List[Dict[str, Any]] = field(default_factory=list)
    excluded: Dict[str, str] = field(default_factory=dict)  # column -> stated reason

    # -- accessors -----------------------------------------------------
    @property
    def batches(self) -> List[str]:
        return list(self.wide.index)

    @property
    def n_batches(self) -> int:
        return len(self.wide.index)

    @property
    def formulations(self) -> pd.Series:
        return self.metadata.loc[self.wide.index, "formulation"]

    @property
    def n_formulations(self) -> int:
        return int(self.formulations.nunique())

    def columns_in(self, *families: str) -> List[str]:
        return [
            c
            for c in self.wide.columns
            if self.families.get(c) in families and c not in self.excluded
        ]

    def predictor_columns(self) -> List[str]:
        return self.columns_in(*PREDICTOR_FAMILIES)

    def response_columns(self) -> List[str]:
        return self.columns_in("RESPONSE")

    def predictors(self) -> pd.DataFrame:
        return self.wide[self.predictor_columns()].astype(float)

    # -- explicit imputation ------------------------------------------
    def missingness(self) -> pd.DataFrame:
        miss = self.wide.isna()
        out = pd.DataFrame(
            {
                "n_missing": miss.sum(),
                "fraction_missing": miss.mean(),
                "family": [self.families.get(c, "UNTAGGED") for c in self.wide.columns],
            }
        )
        return out.sort_values("fraction_missing", ascending=False)

    def impute(self, method: str, columns: Optional[Iterable[str]] = None) -> "BatchDataset":
        """Explicit, recorded imputation. There is no default and no automatic call."""
        allowed = {"median", "mean", "drop_columns", "drop_batches"}
        if method not in allowed:
            raise ValueError(f"method must be one of {sorted(allowed)}")
        cols = list(columns) if columns is not None else [
            c for c in self.wide.columns if self.wide[c].isna().any()
        ]
        new = self.wide.copy()
        if method in ("median", "mean"):
            for c in cols:
                if not pd.api.types.is_numeric_dtype(new[c]):
                    raise ValueError(f"Cannot {method}-impute non-numeric column {c!r}")
                value = new[c].median() if method == "median" else new[c].mean()
                n = int(new[c].isna().sum())
                if n:
                    new[c] = new[c].fillna(value)
                    self.imputation_record.append(
                        {"column": c, "method": method, "value": float(value), "n_filled": n}
                    )
        elif method == "drop_columns":
            new = new.drop(columns=cols)
            for c in cols:
                self.excluded[c] = "dropped by user (missing values)"
                self.imputation_record.append({"column": c, "method": "drop_column"})
        else:  # drop_batches
            keep = ~new[cols].isna().any(axis=1)
            dropped = list(new.index[~keep])
            new = new.loc[keep]
            self.imputation_record.append({"method": "drop_batches", "batches": dropped})
        self.wide = new
        return self

    def exclude(self, column: str, reason: str) -> None:
        """Exclusion always carries a stated reason; nothing is dropped silently."""
        self.excluded[column] = reason


# ---------------------------------------------------------------------
# Build + validate
# ---------------------------------------------------------------------
def _coerce_check(df: pd.DataFrame, families: Dict[str, str], report: ValidationReport):
    """Report non-numeric contamination in columns whose family should be numeric.

    Values are NOT coerced. The offending column keeps its object dtype and is
    excluded from modelling until the user fixes the source data.
    """
    bad = {}
    for col in df.columns:
        fam = families.get(col)
        if fam is None or fam == "META":
            continue
        s = df[col]
        if pd.api.types.is_numeric_dtype(s):
            continue
        converted = pd.to_numeric(s, errors="coerce")
        offending = s[converted.isna() & s.notna()]
        if len(offending):
            bad[col] = sorted({str(v) for v in offending.unique()})[:5]
    for col, examples in bad.items():
        report.add(
            "ERROR",
            "NON_NUMERIC_IN_NUMERIC_COLUMN",
            f"Column {col!r} is tagged {families[col]} but contains non-numeric values "
            f"({', '.join(examples)}). Not coerced; excluded until corrected.",
            [col],
            examples=examples,
        )
    return bad


def build_dataset(
    wide: pd.DataFrame,
    families: Dict[str, str],
    metadata: pd.DataFrame,
    dissolution: Optional[pd.DataFrame] = None,
    batch_col: str = "batch",
) -> BatchDataset:
    """Assemble and validate. Returns a dataset whose .report must be read."""
    report = ValidationReport()

    wide = wide.copy()
    metadata = metadata.copy()

    for name, df in (("batch table", wide), ("metadata", metadata)):
        if batch_col not in df.columns:
            raise ValueError(f"{name} has no {batch_col!r} column")

    # -- duplicate batch ids ------------------------------------------
    for name, df in (("batch table", wide), ("metadata", metadata)):
        dup = df[batch_col][df[batch_col].duplicated()].unique()
        if len(dup):
            report.add(
                "ERROR",
                "DUPLICATE_BATCH_ID",
                f"Duplicate batch identifiers in {name}: {', '.join(map(str, dup))}. "
                "Refusing to merge or aggregate them.",
                detail_batches=list(map(str, dup)),
            )
            # keep first occurrence so downstream code can still run for reporting
            df.drop_duplicates(subset=[batch_col], keep="first", inplace=True)

    wide = wide.set_index(batch_col)
    metadata = metadata.set_index(batch_col)
    wide.index = wide.index.astype(str)
    metadata.index = metadata.index.astype(str)

    if "formulation" not in metadata.columns:
        raise ValueError("metadata must contain a 'formulation' column")

    # -- unknown families / untagged columns --------------------------
    unknown = {c: f for c, f in families.items() if f not in FAMILIES}
    if unknown:
        raise ValueError(f"Unknown column families: {unknown}. Allowed: {FAMILIES}")
    untagged = [c for c in wide.columns if c not in families]
    if untagged:
        report.add(
            "WARNING",
            "UNTAGGED_COLUMNS",
            f"{len(untagged)} column(s) have no family tag and will not be used as "
            "predictors or responses until tagged.",
            untagged,
        )

    # -- cross-table membership ---------------------------------------
    in_wide_not_meta = sorted(set(wide.index) - set(metadata.index))
    in_meta_not_wide = sorted(set(metadata.index) - set(wide.index))
    if in_wide_not_meta:
        report.add(
            "ERROR",
            "BATCH_MISSING_METADATA",
            f"{len(in_wide_not_meta)} batch(es) in the batch table have no metadata row: "
            f"{', '.join(in_wide_not_meta[:10])}",
            detail_batches=in_wide_not_meta,
        )
    if in_meta_not_wide:
        report.add(
            "WARNING",
            "METADATA_WITHOUT_BATCH",
            f"{len(in_meta_not_wide)} metadata row(s) have no batch data: "
            f"{', '.join(in_meta_not_wide[:10])}",
            detail_batches=in_meta_not_wide,
        )

    # -- non-numeric contamination ------------------------------------
    bad_cols = _coerce_check(wide, families, report)

    # -- constant and missing -----------------------------------------
    constant, high_missing, all_missing = [], [], []
    for col in wide.columns:
        s = wide[col]
        frac_missing = float(s.isna().mean())
        if frac_missing == 1.0:
            all_missing.append(col)
            continue
        if frac_missing > THRESHOLDS["missing_fraction_flag"]:
            high_missing.append(col)
        if s.dropna().nunique() <= 1:
            constant.append(col)
    if constant:
        report.add(
            "WARNING",
            "CONSTANT_COLUMN",
            f"{len(constant)} column(s) take a single value across all batches and carry "
            "no information about between-batch differences. Excluded, with reason recorded.",
            constant,
        )
    if high_missing:
        report.add(
            "WARNING",
            "HIGH_MISSINGNESS",
            f"{len(high_missing)} column(s) exceed "
            f"{THRESHOLDS['missing_fraction_flag']:.0%} missing. Not imputed: choose a "
            "method explicitly if you want them used.",
            high_missing,
        )
    if all_missing:
        report.add(
            "WARNING",
            "ALL_MISSING_COLUMN",
            f"{len(all_missing)} column(s) are entirely missing. Excluded.",
            all_missing,
        )

    ds = BatchDataset(
        wide=wide,
        families={c: f for c, f in families.items() if c in wide.columns},
        metadata=metadata.reindex(wide.index),
        dissolution=None,
        report=report,
    )
    for col in constant:
        ds.exclude(col, "constant across all batches - no information to attribute")
    for col in all_missing:
        ds.exclude(col, "no observed values")
    for col in bad_cols:
        ds.exclude(col, "non-numeric values in a numeric column - not coerced")

    # -- dissolution ---------------------------------------------------
    if dissolution is not None:
        ds.dissolution = validate_dissolution(dissolution, ds, report)

    # -- size notes ----------------------------------------------------
    n = ds.n_batches
    p = len(ds.predictor_columns())
    report.add(
        "INFO",
        "SHAPE",
        f"{n} batches, {ds.n_formulations} formulations, {p} usable candidate predictors.",
        n_batches=n,
        n_predictors=p,
        n_formulations=ds.n_formulations,
    )
    return ds


def validate_dissolution(
    diss: pd.DataFrame, ds: BatchDataset, report: ValidationReport
) -> pd.DataFrame:
    required = {"batch", "medium", "time_h", "percent_released"}
    missing = required - set(diss.columns)
    if missing:
        raise ValueError(f"Dissolution table missing columns: {sorted(missing)}")
    diss = diss.copy()
    diss["batch"] = diss["batch"].astype(str)
    unknown = sorted(set(diss["batch"]) - set(ds.wide.index))
    if unknown:
        report.add(
            "WARNING",
            "DISSOLUTION_UNKNOWN_BATCH",
            f"{len(unknown)} batch(es) in the dissolution table are not in the batch table: "
            f"{', '.join(unknown[:10])}",
            detail_batches=unknown,
        )
    absent = sorted(set(ds.wide.index) - set(diss["batch"]))
    if absent:
        report.add(
            "INFO",
            "NO_DISSOLUTION_FOR_BATCH",
            f"{len(absent)} batch(es) have no dissolution data.",
            detail_batches=absent,
        )
    for col in ("time_h", "percent_released"):
        if not pd.api.types.is_numeric_dtype(diss[col]):
            report.add(
                "ERROR",
                "DISSOLUTION_NON_NUMERIC",
                f"Dissolution column {col!r} is not numeric. Not coerced.",
                [col],
            )
    if np.isfinite(diss["percent_released"].to_numpy(dtype=float, na_value=np.nan)).any():
        out_of_range = diss[(diss["percent_released"] < -1) | (diss["percent_released"] > 130)]
        if len(out_of_range):
            report.add(
                "WARNING",
                "DISSOLUTION_OUT_OF_RANGE",
                f"{len(out_of_range)} dissolution value(s) outside -1 to 130 % released.",
            )
    return diss
