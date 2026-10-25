# Interpreting the measurements

Completeness, contamination, taxonomy and RNA gene content answer different
questions. A well-classified genome can still be contaminated; an incomplete
assembly can contain useful biological information. Keep tool outcomes alongside
measurements and choose selection criteria appropriate to the intended analysis.

## Qscore and representative variants

The current composite score is:

```text
Qscore = completeness - 5 × contamination
         + contamination × strain_heterogeneity / 100
         + 0.5 × ln(N50)
```

CheckM percentages are on the 0–100 scale; N50 is in base pairs. The assembly-size
coefficient is zero. Missing required metrics produce an unavailable score.
The score ranks contig-length variants of the same input genome, with N50 and
assembly length as tie-breakers. It is a workflow-specific ranking, not a calibrated
probability that a genome is correct.

## rRNA consistency

The current helper compares BLAST hits between matching rRNA types within each
variant. Its consistency flag requires the minimum reported identity to be at
least 97% and the minimum bit score at least 100. A successfully completed check
with no hits is non-excluding; this is not evidence that an rRNA operon is present
or complete. Failed prediction/comparison receives a missing assessment rather
than a successful no-hit result. Inspect the RNA counts and tool statuses with the
flag. `--require_rrna_consistency` is disabled by default.

## Atlas and selection

Atlas annotations summarize multiple evidence types. Quality tiers and derived
scores are descriptive and do not replace the user's final acceptance criteria.
Metrics requiring failed measurements remain unavailable. The atlas is based on
all assessed representatives, including those outside the selected subset.
Optional labelled-set comparisons and ANI-based selections have their own outputs;
these are distinct from choosing one contig-length variant per input genome.
Cohort ANI groups and domain-specific marker trees run by default; see
[ANI and phylogeny](phylogeny.md) for thresholds and containment semantics.
Labelled FastANI comparison work depends on the supplied comparison configuration; an
unlabelled input directory is not automatically a replicated method comparison.

## Comparisons between labelled sets

The atlas comparison wrapper compares methods across all individual genomes
with an additive Gaussian identity GEE model: `metric ~ method + sample`.
Sample is a categorical fixed effect that adjusts baseline differences; no sample
contrasts are reported. One omnibus method test and one test per method pair are
reported for each metric. The model assumes a common additive method effect
across samples. It uses working independence and a sandwich covariance clustered
by known source anchor, with sample included in the anchor identity. Cognate xPGs
share a cluster with their SAG/MAG anchor; unmatched genomes remain included.
Binary outcomes use the same linear mean model to estimate adjusted prevalence
differences. This does not model arbitrary residual dependence within a sample
or support extrapolation to a population of independently replicated samples.

The independence-assumed sensitivity uses the same sample-adjusted design but
singleton covariance clusters. Overall and pairwise Wald tests receive BH
correction separately by analysis scope and primary/sensitivity analysis.
Each call defines an endpoint family (for example quality metrics or entity
prevalence); corrections do not span output subsets. Entity category-versus-rest
contrasts have their own correction family. Primary columns are `pvalue` and
`qvalue_bh`; sensitivity columns are `pvalue_independent` and
`qvalue_independent_bh`. `sample=all_samples` denotes the pooled test; `samples`
and `n_samples` record the contributing samples. A single-sample subset reduces
to a method-only model.

`group_a_mean` / `group_b_mean` retain observed raw means;
`group_a_adjusted_mean` / `group_b_adjusted_mean` standardize every method to the
same observed sample distribution. Their difference is the reported
`mean_difference_a_minus_b` and is invariant to the common standardization weights
under this additive model. A method completely confounded with sample produces
an explicit skipped status rather than a misleading p-value. Missing sample,
category or metric values are excluded; no genome matching restriction is imposed.

Tests require at least ten anchor groups in every category involved in a
contrast. This is a safeguard, not a guarantee of adequate asymptotic inference,
especially for rare binary features. Constant outcomes, singular contrasts and
unresolved xPG anchors have missing p-values and an explicit status. Clustering
accounts for identified anchor relationships, not every possible biological
relationship between genomes. Neither sensitivity p-values nor an independence
assumption establish independence.

Existing component-level matched-feature sign tests and representative-selection
win tests remain two-sided exact binomial tests, now separated by sample. They
use one component-level direction or decisive winner, exclude ties, and apply BH
within each returned comparison table. They test directional consistency or
selection frequency, not genome-level mean differences. They assume components
are independent; no duplicate independence sensitivity is needed for an already
collapsed component endpoint. Selected-genome results describe the selected
subset and should not be interpreted as an unbiased assessment of selection.
Counts with only one value per sample/category are descriptive.

Pooled panels again show significance brackets using sample-adjusted,
anchor-robust BH q-values. Their captions distinguish mean-contrast inference
from descriptive medians, IQRs or raw means drawn in the panels.
`*_method_effects_all_samples_<metric>.png/pdf` additionally show adjusted mean
differences, unadjusted robust 95% confidence intervals and BH stars. A confidence
interval excluding zero need not imply BH significance. Independence-sensitivity
results remain in tables rather than providing a second set of plot brackets.
Entity differential plots use adjusted prevalence differences; raw prevalence
counts and raw odds ratios remain descriptive table columns.



These statistical options apply to configured comparison analyses. A simple full
run does not create a study design or independent replication from genome files.
MetaPathways integration scripts are outside this guide's complete-workflow scope.
