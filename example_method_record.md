# Method record

This application is a development and investigation aid. Nothing it produces is a validated GMP output.

## Run

- **model_type**: elastic_net
- **cv_scheme**: leave_one_formulation_out
- **n_batches**: 24
- **n_formulations**: 4
- **n_candidate_predictors**: 74
- **n_model_features**: 65
- **settings**: `{'seed': 20240101, 'n_permutations': 1000, 'n_bootstrap': 500, 'inner_cv_splits': 5, 'collinear_cluster_r': 0.9, 'stability_threshold': 0.6, 'permutation_alpha': 0.05, 'extra': {}}`
- **libraries**: `{'python': '3.12.3', 'numpy': '2.4.4', 'pandas': '3.0.2', 'scipy': '1.17.1', 'scikit-learn': '1.8.0'}`

## Inseparable clusters modelled as single features

- `cluster[hardness_N+4]`: hardness_N, hpmc_level_pct, hpmc_mg, main_compression_kN, pre_compression_kN
- `cluster[granulation_torque_Nm+1]`: granulation_torque_Nm, granulation_water_pct

## Columns excluded at load

- `api_pct_ww`: constant across all batches - no information to attribute
- `api_mg`: constant across all batches - no information to attribute
- `total_tablet_weight_mg`: constant across all batches - no information to attribute

## Zero-information columns

- api_mg
- api_pct_ww
- total_tablet_weight_mg

## Removed to break a mixture or linear dependency

- lactose_pct_ww
- mg_stearate_pct_ww
- talc_pct_ww

## Predictors excluded for missingness

- coating_weight_gain_pct

## Design assessment

- 3 column(s) were excluded when the data was loaded and are not candidates here: api_mg (constant across all batches - no information to attribute); api_pct_ww (constant across all batches - no information to attribute); total_tablet_weight_mg (constant across all batches - no information to attribute).
- 24 batches against 74 candidate predictors across 4 formulations.
- p is greater than n/3 (74 > 8.0). Variable selection will be unstable: different resamples of the same data will select different predictors, and any single ranking should be read as one draw from a wide distribution.
- With p >= n (74 >= 24) an unpenalised model fits the data exactly and tells you nothing. Only penalised or latent-variable models are offered, and only against a permutation null.
- Batches within a formulation are not independent replicates of the formulation factor. Effective sample size for anything that changed only between formulations is 4, not 24 (batches per formulation: F1=6, F2=6, F3=6, F4=6).
- Cannot be separated: hardness_N, hpmc_level_pct, hpmc_mg, lactose_pct_ww, main_compression_kN, pre_compression_kN. These move together across all 24 batches (for example r = 1.00 between hpmc_level_pct and hpmc_mg). Any model attributing the response change to one of them rather than another is making an arbitrary choice, not a finding.
- Cannot be separated: granulation_torque_Nm, granulation_water_pct. These move together across all 24 batches (for example r = 0.96 between granulation_torque_Nm and granulation_water_pct). Any model attributing the response change to one of them rather than another is making an arbitrary choice, not a finding.
- 3 predictor(s) are aliased with formulation identity (R-squared > 0.95 on formulation alone): hpmc_level_pct, hpmc_mg, main_compression_kN. Each is a formulation label in disguise and cannot be credited independently of everything else that changed at the same time.
- Mixture constraint detected: hpmc_level_pct, lactose_pct_ww, mcc_pct_ww, mg_stearate_pct_ww, talc_pct_ww sum to a constant (75.00, sd 0.0000). Ordinary regression on these raw components is singular. Fit a mixture model, or omit one component as the dependent remainder; the app will not fit them all together.
- Exact linear dependency among predictors: -0.34*lactose_pct_ww -0.18*hpmc_level_pct -0.03*hpmc_mg -0.34*mcc_pct_ww -0.34*talc_pct_ww -0.34*mg_stearate_pct_ww is constant. These cannot all appear in the same model.
- Exact linear dependency among predictors: +0.06*hpmc_mg -0.30*hpmc_level_pct +0.01*lactose_pct_ww +0.01*mcc_pct_ww +0.01*talc_pct_ww +0.01*mg_stearate_pct_ww is constant. These cannot all appear in the same model.
- What this dataset can address: factors that vary within formulations, where between-batch replication exists. There are 71 of these.
- What it cannot address: the individual contribution of any factor listed above as aliased or inseparable. No amount of modelling recovers information the data does not contain; only a designed experiment does.

## Response assessment

- Between-formulation signal exceeds within-formulation noise (F = 36.58, ICC = 0.86); within-formulation SD 2.9, between-formulation SD 7.05.

## Warnings

- 1 predictor(s) have missing values and were excluded rather than imputed: coating_weight_gain_pct. Choose an imputation method explicitly (it will be recorded), or pass allow_missing_predictors=True to accept the training-mean fallback.
- Variance inflation factors cannot be computed at all: p=74 on the full set and 68 after clustering, both against n-1=23. With more predictors than batches every predictor is an exact linear combination of the others, which is itself the finding.

## Validation

- **scheme**: leave_one_formulation_out
- **q2**: -0.24381205853207066
- **rmse**: 7.510952170059962
- **n_used**: 24
- **permutation_p**: 0.936063936063936
- **permutation_n**: 1000
- **permutation_null_p95**: -0.0067352784534422646
- **random_split_q2_for_comparison_only**: 0.6694249492302143
- **stability_n_effective**: 500
- **verdict**: not_supported

## Conclusion

No attribution supported. With 24 batches and 74 candidate predictors, the best model achieved a cross-validated Q-squared of -0.24 against a permutation null whose 95th percentile is -0.01 (p = 0.94). The apparent relationships in the raw data do not survive held-out validation. 3 predictor(s) are aliased with formulation identity and cannot be assessed independently in any case.

## Proposed confirmatory experiment

```
Proposed design: 3-factor two-level full factorial (2^3) with centre points.
  Vary hpmc_level_pct from 17.3 to 24.8 (observed range 17.3 to 24.8).
  Vary main_compression_kN from 7.5 to 16.7 (observed range 7.5 to 16.7).
  Vary pre_compression_kN from 1.38 to 3.36 (observed range 1.38 to 3.36).
  Hold constant: api_bulk_density_g_ml, api_d10_um, api_d50_um, api_d90_um, api_moisture_pct, api_tapped_density_g_ml, hardness_N (measured, expected to follow), lactose_pct_ww (moves as the mixture remainder).
  10 batches including 2 centre point(s).
  Confirmation: q30_pct must move with hpmc_level_pct at both levels of the other factor(s), with the hpmc_level_pct main effect exceeding the centre-point replicate standard deviation by a factor of three. If the response instead tracks main_compression_kN, the attribution belongs there.
```

## Decision rules applied

- **columns excluded**: constant across all batches, entirely missing, or non-numeric in a numeric family; each with its reason recorded, never dropped silently
- **p > n/3**: flagged as unstable variable selection (threshold 0.333)
- **inseparable cluster**: |r| >= 0.9 on the predictor correlation graph, reported and modelled as one feature
- **aliased with formulation**: R-squared > 0.95 regressing the predictor on formulation identity
- **aliased with time**: R-squared > 0.95 regressing the predictor on manufacturing date
- **mixture constraint**: component percentages summing to a constant within 0.5; one component removed as the dependent remainder
- **complexity cap**: max_features = clip(floor(n_train/5), 1, 10), enforced by moving up the regularisation path inside every training fold
- **cross-validation**: grouped only; leave-one-formulation-out by default. No random row split is reachable for reporting
- **hyperparameter tuning**: inner loop only, refitted inside every outer fold, every bootstrap resample and every permutation
- **variable selection**: refitted inside every resample; never selected once on all data and cross-validated afterwards
- **permutation test**: response shuffled, whole pipeline refitted; empirical p = (1 + #{null >= observed}) / (n + 1); gate at p <= 0.05
- **stability threshold**: selection frequency below 60% labelled UNSTABLE and not treated as a finding
- **non-linear models**: offered below n = 30 only with a warning; depth capped at 3
- **findings gate**: no importance ranking, partial dependence or contribution plot is produced when the permutation test failed

## References

- *False discovery rate on the univariate screen* — Benjamini Y, Hochberg Y. Controlling the false discovery rate. J R Stat Soc B 1995;57:289-300.
- *Elastic net, and why correlated predictors are kept together* — Zou H, Hastie T. Regularization and variable selection via the elastic net. J R Stat Soc B 2005;67:301-320.
- *Stability selection* — Meinshausen N, Buhlmann P. Stability selection. J R Stat Soc B 2010;72:417-473.
- *Permutation testing of cross-validated performance* — Ojala M, Garriga GC. Permutation tests for studying classifier performance. J Mach Learn Res 2010;11:1833-1863.
- *Selection bias from selecting once and cross-validating afterwards* — Ambroise C, McLachlan GJ. Selection bias in gene extraction on the basis of microarray gene-expression data. PNAS 2002;99:6562-6566.
- *Nested cross-validation for honest performance estimates* — Varma S, Simon R. Bias in error estimation when using cross-validation for model selection. BMC Bioinformatics 2006;7:91.
- *Y-scrambling in QSAR-style small-n modelling* — Rucker C, Rucker G, Meringer M. y-Randomization and its variants in QSPR/QSAR. J Chem Inf Model 2007;47:2345-2357.
- *PLS for correlated pharmaceutical predictors* — Wold S, Sjostrom M, Eriksson L. PLS-regression: a basic tool of chemometrics. Chemom Intell Lab Syst 2001;58:109-130.
- *Mixture constraints in formulation data* — Cornell JA. Experiments with Mixtures. 3rd ed. Wiley, 2002.
- *Dissolution profile comparison and f2* — FDA Guidance for Industry: Dissolution Testing of Immediate Release Solid Oral Dosage Forms, 1997; EMA CHMP/EWP/QWP/1401/98 Rev.1.
- *Model-independent profile metrics (MDT, dissolution efficiency)* — Costa P, Lobo JMS. Modeling and comparison of dissolution profiles. Eur J Pharm Sci 2001;13:123-133.
- *Content uniformity acceptance value* — USP <905> Uniformity of Dosage Units.
- *Status of this application under GMP* — EU GMP Annex 22 (draft, 2025): models in critical applications must be static and deterministic with documented human oversight and change control. This application is a development and investigation aid, not a validated GMP output.