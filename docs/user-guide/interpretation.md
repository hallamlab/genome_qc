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

## Integrity and recoverability

These two atlas scores describe complementary aspects of the **assembled genome**:
its marker-based completeness and contamination, and the RNA gene features recovered
in its sequence. Both range from 0 to 1; neither is a probability or a percentage
of the genome recovered.

### Integrity: how favorable is the completeness–contamination balance?

Integrity summarizes whether an assembly contains much of the expected genome
content while showing little marker-based evidence of contamination. It uses
CheckM completeness and contamination and takes the **weaker of the two components**.
A high completeness estimate therefore cannot compensate for high contamination,
and low contamination cannot compensate for substantial missing sequence.

The completeness component increases from 0 at 50% completeness to 1 at 90%.
The contamination component decreases from 1 at 5% contamination to 0 at 10%.
Values beyond these ranges are capped. For example:

- A genome with 70% completeness and 2% contamination has integrity **0.5**,
  limited by completeness.
- A genome with 95% completeness and 7.5% contamination also has integrity
  **0.5**, but is limited by contamination.
- Integrity **1** means completeness is at least 90% and contamination is at
  most 5%. It does not distinguish a 90%-complete assembly from a 99%-complete one.

Always inspect the underlying measurements to understand a low or intermediate
score. Integrity does not assess every possible assembly error or establish that
all contigs belong to the same organism. Contiguity, RNA features and GUNC findings
provide separate evidence and are not included in this score.

### Recoverability: which RNA gene hallmarks did the assembly recover?

Recoverability describes the representation of RNA genes involved in the cell's
translation machinery: **16S, 23S and 5S rRNA, plus a tRNA diversity criterion**.
Despite its name, it measures features already recovered in this assembly, not
how likely further sequencing or reassembly is to recover more of the genome.

Each of the four components contributes up to one quarter of the score. Each
rRNA component uses its weighted count, capped at 1; the tRNA component is 1 when
at least 18 distinct predicted tRNA isotype labels are present and 0 otherwise.
Thus, three fully scored rRNA components with the tRNA criterion unmet give
**0.75**. Two fully scored rRNA components plus the tRNA criterion also give
**0.75**. Inspect the individual RNA columns or recovery pattern to see which
features account for the score.

Partial rRNA predictions contribute fractional counts, and several partial
predictions can add up to a fully scored component. A score of 1 therefore does
not by itself establish intact full-length rRNA genes, a linked rRNA operon, or
a complete chromosome. Conversely, a missing predicted feature does not establish
that the organism lacks it: the sequence may be absent from the assembly or not
recognized by the prediction step. This score does not measure gene expression,
cellular activity or metabolic capability.

### Reading the two scores together

In the integrity-versus-recoverability plot, integrity is on the horizontal axis
and recoverability on the vertical axis. Use their combination to decide what to
inspect next:

| Score pattern | Biological reading and next check |
| --- | --- |
| High integrity, high recoverability | Favorable marker-based quality with recovered RNA hallmarks. Inspect raw quality metrics and other assessments before selecting the genome. |
| High integrity, low recoverability | Favorable marker-based quality, but some RNA hallmarks are absent or partially recovered. Check the specific RNA features, especially if the intended analysis depends on them. |
| Low integrity, high recoverability | RNA hallmarks are represented, but completeness or contamination remains limiting. RNA recovery does not resolve that limitation; inspect CheckM and GUNC results. |
| Low integrity, low recoverability | Both the marker-based quality profile and recovered RNA features are limited. The genome may still support specific gene-level observations, but absence claims require caution. |

“High” and “low” here are reading aids, not additional selection thresholds.
Missing assessments are unavailable rather than evidence of biological absence.
The scores are descriptive: they do not automatically exclude genomes under the
default settings. Integrity of 1 alone also does not establish a high-quality
tier, which has additional RNA requirements and stricter completeness and
contamination boundaries. See the [atlas column definitions](output-columns.md#atlas-derived-columns)
for exact formulas, recovery patterns and tier rules.

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
