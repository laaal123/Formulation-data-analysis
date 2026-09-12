> **This is the FLAT single-folder build.** Every Python file sits at the top level, so
> you can select all and upload in one action. It is functionally identical to the
> packaged build and passes the same 99 tests, with three differences:
>
> - imports are `from modelling import ...` rather than `from batchattr.modelling import ...`
> - the tests live beside the source instead of in `tests/`
> - `.streamlit/config.toml` (theme) and `.github/workflows/ci.yml` (CI) are omitted,
>   because they only work inside their dot-folders. Neither affects whether the app runs;
>   add them later via **Add file -> Create new file** if you want them.
>
> If you would rather have the conventional package layout, use the other build.

# Batch factor attribution

Takes formulation, process and IPQC data across batches and identifies which factors
are **associated** with a chosen response — then names everything each factor cannot
be separated from, and states the experiment that would settle it.

A formulation series is not a designed experiment, and this data has p ≫ n. The honest
output is therefore a ranked list of hypotheses with their confounders attached, or a
refusal. The app is built to produce both, and to refuse loudly.

**Nothing produced here is a validated GMP output.** It is a development and
investigation aid. Under draft EU GMP Annex 22, models in critical applications must be
static and deterministic with documented human oversight and change control; this is
built deterministically (fixed seeds, pinned versions, a logged method record) so that
the same input always gives the same output.

---

## Install and run

```bash
pip install -r requirements-dev.txt   # runtime + pytest
# or: pip install -r requirements.txt  # runtime only, what Streamlit Cloud installs

streamlit run app.py          # the interface
python run_example.py         # command-line demo on synthetic data with known truth
pytest -q                     # 99 tests, must be zero failures
```

`run_example.py --record method_record.md` writes the full method record.

**Deploying:** see `UPLOAD_ORDER.md` for the file-by-file upload sequence for GitHub and
the Streamlit Community Cloud settings. Entry point is `app.py` at the repository root;
Streamlit Cloud installs `requirements.txt`.

---

## Interface

Eight tabs in the order the analysis must be done in, and they are **gated**:

| # | Tab | Unlocked when |
|---|---|---|
| 1 | Data | always |
| 2 | Design diagnostics | data loaded |
| 3 | Response | diagnostics have been viewed |
| 4 | Model | the chosen response passed its own checks |
| 5 | Validation | a model has been configured |
| 6 | **Findings** | **validation ran and the permutation test passed** |
| 7 | Next experiment | validation ran (either way) |
| 8 | Method record | data loaded |

Findings carries a persistent banner: *Association from observational batch data. Not
causation. See the confound list for each factor.*

Gating lives in `gating.py`, outside Streamlit, so it is unit-tested.

---

## Modules

| File | Prompt | Contents |
|---|---|---|
| `loader.py` | 1 | wide/long ingestion, dissolution and metadata tables, `ValidationReport`, explicit imputation |
| `synthetic.py` | 1 | 24 batches × 4 formulations with confounding built to known ground truth; controlled generators for testing |
| `diagnostics.py` | 2 | design audit, cannot-separate clusters, VIF, formulation/time aliasing, mixture detection, range and replication |
| `response.py` | 3 | scalar responses; profile reduction to Weibull, first-order k, MDT, DE, f2, t50/t80, cross-medium spread; signal-vs-noise and method-precision gates |
| `modelling.py` | 4 | univariate screen with BH FDR, elastic net / LASSO / ridge / PLS / GBM / RF, cluster scores, complexity cap, stability selection |
| `validation.py` | 5 | grouped and nested CV, permutation test, Q², verdict |
| `interpret.py` | 6 | findings with confound lists, gated dependence and contribution plots, confirmatory design generator, causal-language linter |
| `gating.py`, `record.py` | 7 | tab gates; method record with decision rules and references |
| `pipeline.py` | — | orchestrator enforcing the order |

---

## The model ladder

| Model | Why it is there |
|---|---|
| Univariate screen (BH FDR) | A filter, never a finding. Its p-values are forbidden from the output |
| **Elastic net** (default) | Handles p >> n and keeps correlated groups together |
| LASSO | Sparser, but splits a confounded pair with no guarantee which twin survives |
| **Group lasso** | An inseparable cluster enters or leaves as a whole. Custom solver: orthonormalised blocks, covariance updates, active set |
| Ridge | Keeps everything, shrinks everything. Closed form via one SVD |
| PLS | Latent components from correlated predictors |
| **Scheffe mixture** (linear, quadratic) | Models the simplex with no intercept, instead of deleting a component. Coefficients are contrasts between components |
| GBM, random forest | Only with a reason. Depth capped at 3, warned below n = 30 |

For models with no zero coefficients (ridge, PLS, Scheffe, GBM, RF) "selected" means inside
the complexity cap by absolute effect, and is labelled that way. Otherwise stability
selection would report 100% for every feature and mean nothing.

**A warning about the Scheffe model.** It fits the mixture components alone, so it can pass
validation while every process parameter sits outside the model. On the example data it
reports Q² = 0.44 and p = 0.016 for HPMC level — but compression force was never in the
race. The headline says so explicitly, and the confound list still comes from the design
diagnostics rather than from the model.

## What the app refuses to do

- Fit anything before the design assessment has been produced.
- Report a random train/test split. `make_splits` offers grouped schemes only; the
  random number is computed in exactly one place, labelled, purely to show the user how
  much optimism it would have bought.
- Tune hyperparameters outside the outer loop, or select variables once and
  cross-validate afterwards. Clustering statistics, standardisation, tuning and
  selection are all refit inside every fold, every bootstrap and every permutation.
- Report a training R². It is never computed.
- Show an importance ranking, partial dependence plot or contribution table when the
  permutation test failed. Those functions raise `PermissionError`.
- Impute anything silently. Missing predictors are excluded with a stated reason unless
  you opt in, and every imputation is recorded.
- Fit raw mixture components. One is removed as the dependent remainder.
- Exceed `max_features = clip(floor(n_train/5), 1, 10)`, enforced per training fold.
- Use causal language. A linter checks the app's own output and the test suite fails on
  a hit.

---

## What good output looks like

The synthetic example is built the way a real formulation series is built: HPMC level,
compression force and hardness all moved together between F1 and F4. The app says so.

```
No attribution supported. With 24 batches and 74 candidate predictors, the best model
achieved a cross-validated Q² of -0.24 against a permutation null whose 95th percentile
is -0.04 (p = 0.93). The apparent relationships in the raw data do not survive held-out
validation. 3 predictor(s) are aliased with formulation identity.

For comparison only: a random row split on the same data gives Q² 0.669 against the
grouped -0.244.

Cannot be separated: hardness_N, hpmc_level_pct, hpmc_mg, lactose_pct_ww,
main_compression_kN, pre_compression_kN (r = 0.97 between hpmc_level_pct and
main_compression_kN).

Proposed design: 3-factor two-level full factorial with centre points.
  Vary hpmc_level_pct from 17.3 to 24.8
  Vary main_compression_kN from 7.5 to 16.7
  Vary pre_compression_kN from 1.38 to 3.36
  Hold constant: ... hardness_N (measured, expected to follow),
                 lactose_pct_ww (moves as the mixture remainder)
  10 batches including 2 centre points.
```

The ground truth is that HPMC level is the driver. The app is right not to say so: the
data cannot distinguish it from compression force, and a ranking that named it would
have been luck.

---

## Tests

```
tests/test_correctness.py   35   known driver recovered; pure noise refused;
                                 confounded pair kept together; grouped CV audited;
                                 loader validation; profile metrics against analytic
                                 values; determinism
tests/test_robustness.py     6   14 degenerate scenarios × 9 methods, each classified
                                 OK / REFUSED / CRASH / SILENT WRONG ANSWER;
                                 crashes and silent wrong answers must be zero, and
                                 every refusal is audited as genuine
tests/test_interface.py     17   tab gating, method record completeness, references
tests/test_guardrails.py    15   the ten failure modes from the guardrail prompt,
                                 checked in source and in behaviour
```

The single most important test is `test_pure_noise_is_refused`. A tool that finds
drivers in noise is worse than no tool.

---

## Known limits

- Leave-one-formulation-out asks whether you can predict a **new** formulation. For a
  factor that only moved between formulations that is extrapolation, and a negative Q²
  is the correct answer rather than a bug. Leave-one-batch-out is offered, with a
  warning that it answers a different and easier question.
- Cluster scores assume the members of an inseparable group move together. They always
  have in this data — that is why they are inseparable — but a per-unit effect quoted
  for one member is conditional on that.
- With four formulations, leave-one-formulation-out gives four folds. Everything from
  such a design is indicative, and the app says so.
- SHAP for tree models is not bundled. For linear models the contributions reported are
  the exact Shapley values, computed in closed form.
