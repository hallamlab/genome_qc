# Output table column reference

Use this dictionary alongside [results and explorer](outputs.md),
[measurement interpretation](interpretation.md), and [ANI and phylogeny](phylogeny.md).
Names are case-sensitive: for example, `Genome_Id` and `genome_id` are different
headers even when they identify the same genome. Definitions below follow the
current workflow and its published tool tables. Optional comparisons can add
metadata columns chosen by the user; those retain the meaning of the supplied
metadata, not a meaning inferred by Genome QC.

## Reading tables safely

- **Row unit matters.** The master has one assessed representative per original
  genome; individual tool tables can have one row per contig-length variant,
  gene, hit, genome pair, group or statistical contrast.
- **Missing is not zero.** Empty cells, `NA`, `N/A` or `NaN` can denote an unavailable
  measurement. Check the assessment/status and reason columns. A zero RNA count
  from a successful prediction means no feature was detected, not that prediction
  failed. Boolean fields may be written as `True`/`False` or `1`/`0`.
- **Percent versus fraction:** CheckM completeness/contamination and ANI use
  percent (0–100, although estimated contamination can exceed 100). AF and most
  derived atlas scores use fractions (0–1). Sequence lengths are base pairs (bp).
- **IDs are labels, not taxonomy.** `Bin Id` identifies the representative variant,
  commonly `<Genome_Id>.full`; `.full` means no contig-length threshold was applied,
  not a complete chromosome. Numeric suffixes identify filtered variants.
- **Selection is separate from assessment.** With no acceptance thresholds, all
  assessed representatives can be selected even if quality is poor or a
  measurement is unavailable. Use both `selected` and `assessment_complete`.

## Input inventory and identifiers

`Genome_inventory.tsv` records every input, including inputs not represented in
an assessment table. `inputs.resolved.json` retains the original manifest fields
plus checksums. Arbitrary extra manifest columns remain user metadata.

| Column | Type / units | Meaning |
| --- | --- | --- |
| `Genome_Id` | Text | Stable original input identifier; join key for final genome-level outputs. |
| `Bin Id` | Text | Identifier of the selected contig-length variant; join key for variant-level tool results. |
| `sample` | Text / blank | Optional sample label supplied in the manifest. |
| `category` | Text / blank | Optional grouping label supplied in the manifest; not an inferred taxon. |
| `input_fasta` | Path | Original input FASTA path, not the disposable staged copy. |
| `input_sha256` | Hex string | SHA-256 of the original file bytes; a compressed input is hashed as supplied. |
| `status` | Enum | Inventory: `assessed_selected`, `assessed_not_selected`, or `not_assessed`. Other tables use their own status vocabularies below. |
| `selection_reasons` | Text | `passed`, or semicolon-separated acceptance criteria that failed. For an unassessed input the inventory records why no representative was available. |

Legacy reports may contain `included_in_atlas`, `excluded_after_rRNA_check`, or
`excluded_by_size_or_quality`; the explorer recognizes them, but they do not
represent the current assess-every-input selection workflow.

## Master and selected genome tables

`Master_genome_QC.tsv` contains all assessed representatives.
`Selected_genome_QC.tsv` has the same columns restricted to `selected=True`.
The atlas annotated table and tree-tip metadata inherit these columns.

### Assembly and CheckM measurements

| Column | Type / units | Meaning |
| --- | --- | --- |
| `Completeness` | Percent | CheckM lineage-marker estimate of recovered genome content; not a reference-alignment coverage measurement. |
| `Contamination` | Percent | CheckM marker-copy estimate of contamination/redundancy; not the fraction of contaminant reads. |
| `Strain heterogeneity` | Percent | CheckM estimate of closely related strain contributions among multicopy markers; interpret with contamination. |
| `num_seqs` | Integer count | Number of sequences/contigs in the representative FASTA. |
| `sum_len` | bp | Sum of representative sequence lengths. |
| `min_len` | bp | Shortest representative sequence. |
| `avg_len` | bp | Arithmetic mean sequence length. |
| `max_len` | bp | Longest representative sequence. |
| `N50` | bp | Sequence length at which sequences of that length or longer account for at least half the total assembly length. |
| `qscore` | Dimensionless score | Variant-ranking score: `Completeness − 5×Contamination + Contamination×Strain heterogeneity/100 + 0.5×ln(N50)`. Higher ranks first; it is not a probability or percent. Missing required inputs leave the score unavailable. |

### RNA features and taxonomy

| Column | Type / units | Meaning |
| --- | --- | --- |
| `pass_BARRNAP` | Boolean / missing | rRNA self-comparison consistency result: minimum reported matching-subunit BLAST identity ≥97% and minimum bit score ≥100. A successfully completed check with no eligible hits is non-excluding; it does not establish rRNA recovery. |
| `16S_rRNA` | Weighted count | Sum of parsed barrnap 16S feature contributions: 1 for a feature without a partial-alignment note, otherwise reported aligned percent /100. Can be fractional. |
| `23S_rRNA` | Weighted count | Sum of 23S feature contributions using the same partial-alignment weighting. |
| `5S_rRNA` | Weighted count | Sum of 5S feature contributions using the same partial-alignment weighting. |
| `trna_total` | Count | Total parsed tRNA features, including repeated isotypes. |
| `trna_unique` | Count | Number of distinct parsed tRNA isotype labels; not distinct anticodons or total gene copies. |
| `contains_16S` | Boolean / missing | Whether the representative has a detected 16S feature; prediction failure is unavailable. |
| `Domain` | Text / blank | GTDB domain, with the `d__` prefix removed. |
| `Phylum` | Text / blank | GTDB phylum, with the `p__` prefix removed. |
| `Class` | Text / blank | GTDB class, with the `c__` prefix removed. |
| `Order` | Text / blank | GTDB order, with the `o__` prefix removed. |
| `Family` | Text / blank | GTDB family, with the `f__` prefix removed. |
| `Genus` | Text / blank | GTDB genus, with the `g__` prefix removed. |
| `Species` | Text / blank | GTDB species, with the `s__` prefix removed. An unnamed rank is not proof of a novel species. |
| `rrna_domain_basis` | Text | `GTDB-Tk` when a domain was available; otherwise `Bacteria fallback; taxonomy unavailable`. Records model selection, not independent taxonomic evidence. |
| `trna_domain_basis` | Text | `GTDB-Tk` when a domain was available; otherwise `general model; taxonomy unavailable`. |

### GUNC measurements

The master prefixes GUNC fields with `gunc_`. Raw
`GUNC.*.maxCSS_level.tsv` has one row per genome at the selected maximum-CSS
rank and uses the unprefixed names in parentheses below. GUNC assesses
taxonomic inconsistency across contigs; it is not the same estimate as CheckM
contamination. The source `genome` column corresponds to `Bin Id`.

| Master column (raw name) | Units | Meaning |
| --- | --- | --- |
| `gunc_n_genes_called` (`n_genes_called`) | Count | Predicted genes considered by GUNC. |
| `gunc_n_genes_mapped` (`n_genes_mapped`) | Count | Genes with retained reference-search mappings. |
| `gunc_n_contigs` (`n_contigs`) | Count | Contigs represented in GUNC's scored gene assignments. |
| `gunc_taxonomic_level` (`taxonomic_level`) | Rank | Taxonomic rank of the reported score. |
| `gunc_proportion_genes_retained_in_major_clades` (`proportion_genes_retained_in_major_clades`) | Fraction | Retained major-clade genes divided by mapped genes. |
| `gunc_genes_retained_index` (`genes_retained_index`) | Fraction | Major-clade genes divided by called genes. |
| `gunc_clade_separation_score` (`clade_separation_score`) | 0–1 | CSS: separation of gene taxonomy among contigs; high values suggest chimerism when reference representation is adequate. |
| `gunc_contamination_portion` (`contamination_portion`) | Fraction | Retained genes outside the most abundant clade. |
| `gunc_n_effective_surplus_clades` (`n_effective_surplus_clades`) | Dimensionless | Inverse-Simpson effective clade count minus one. |
| `gunc_mean_hit_identity` (`mean_hit_identity`) | Fraction | Mean reference identity for genes in retained major clades. |
| `gunc_reference_representation_score` (`reference_representation_score`) | 0–1 | RRS: `genes_retained_index × mean_hit_identity`; low values limit interpretation. |
| `gunc_pass` (`pass.GUNC`) | Boolean | GUNC's own pass flag, not Genome QC's final selection or a guarantee of non-chimerism. |

See the [upstream GUNC output reference](https://grp-bork.embl-community.io/gunc/output.html)
for algorithm-specific definitions. Genome QC's additional interpretations are below.

### Assessment completion and final selection

| Column | Type | Meaning |
| --- | --- | --- |
| `checkm_status` | Enum | CheckM outcome for this representative. |
| `gtdbtk_status` | Enum | GTDB classification outcome. |
| `rrna_status` | Enum | rRNA prediction outcome. |
| `rrna_check_status` | Enum | rRNA consistency-comparison outcome. |
| `trna_status` | Enum | tRNA prediction outcome. |
| `gunc_status` | Enum | GUNC outcome. |
| `sequence_stats_status` | Enum | Sequence statistics availability. |
| `assessment_complete` | Boolean | All six tool assessments and sequence statistics are `completed`. This is not a biological quality judgment. |
| `pass_genome_size` | Boolean | Assembly size meets the enabled inclusive minimum/maximum bounds. True if neither is set. |
| `pass_completeness` | Boolean | Completeness meets the enabled inclusive minimum; true if disabled. |
| `pass_contamination` | Boolean | Contamination meets the enabled inclusive maximum; true if disabled. |
| `pass_qscore` | Boolean | Qscore meets the enabled inclusive minimum; true if disabled. |
| `pass_rRNA_check` | Boolean | Requires `pass_BARRNAP` only when `--require_rrna_consistency` is enabled; otherwise true. |
| `selected` | Boolean | All five acceptance checks above pass. Failed/missing values fail an enabled numerical criterion. |
| `selection_reasons` | Text | Failed checks without their `pass_` prefix, separated by semicolons, or `passed`. |

`Assessment_status.tsv` contains the detailed audit, one row per representative
and assessment stage:

| Column | Meaning |
| --- | --- |
| `Genome_Id` | Original input ID. |
| `variant` | Assessed `Bin Id`. |
| `stage` | `checkm`, `gtdbtk`, `rrna`, `rrna_check`, `trna`, `gunc`, or `sequence_stats`. |
| `status` | `completed`: measurement available; `failed`: tool/task failure captured; `blocked`: prerequisites unavailable; `not_completed`: no completion receipt; `not_reported`: task completed but did not report that genome/required metric. |
| `exit_code` | Captured process exit code, or blank when not applicable/available. |
| `reason` | Explanation of a missing or failed assessment; inspect this before interpreting blanks. |
| `log` | Relative published diagnostic log path when recorded. |
| `retry_token` | Recovery/cache invalidation token, not a count of retries. |

## Atlas-derived columns

`genome_quality_atlas/<prefix>_annotated.tsv` adds these to the master table.
The default prefix is `genome_quality`. These are descriptive workflow scores,
not calibrated probabilities or substitutes for direct inspection of genomes.
For biological interpretation and worked examples, see
[Integrity and recoverability](interpretation.md#integrity-and-recoverability).

| Column | Units | Definition |
| --- | --- | --- |
| `rrna_16S_score` | 0–1 | Weighted 16S count clipped to [0,1]; partial recovery can remain fractional. |
| `rrna_23S_score` | 0–1 | 23S count clipped to [0,1]. |
| `rrna_5S_score` | 0–1 | 5S count clipped to [0,1]. |
| `has_16S` | 0/1 | Whether the 16S score is greater than zero. |
| `has_23S` | 0/1 | Whether the 23S score is greater than zero. |
| `has_5S` | 0/1 | Whether the 5S score is greater than zero. |
| `trna_ge_18` | 0/1 | At least 18 distinct tRNA isotype labels were recovered. |
| `mC` | 0–1 | `clip((Completeness − 50)/40, 0, 1)`. |
| `mK` | 0–1 | `clip((10 − Contamination)/5, 0, 1)`. |
| `integrity_score` | 0–1 | Minimum of `mC` and `mK`: the weaker completeness/contamination component. |
| `recoverability_score` | 0–1 | Mean of the three rRNA scores and `trna_ge_18`. Measures hallmark recovery, not expected future assembly improvement. |
| `mimag_quality_index` | 0–1 | Square root of `integrity_score × recoverability_score`. A Genome QC composite, not an official MIMAG score. |
| `recovered_feature_count` | Integer 0–4 | Sum of `has_16S`, `has_23S`, `has_5S`, and `trna_ge_18`. |
| `mimag_tier` | Enum | `high`: completeness >90%, contamination <5%, all three rRNA scores ≥1 and tRNA criterion met; `medium`: completeness ≥50%, contamination <10%, not high; `low`: completeness <50%, contamination <10%; otherwise `failed`. Missing quality/RNA assessments leave this unavailable. This implementation uses weighted feature sums; several partial features can sum to ≥1, so it does not independently establish a full-length rRNA for each subunit. |
| `recovery_pattern` | Four-character string | Presence bits in the order 16S, 23S, 5S, tRNA≥18 (e.g. `1011`); `unavailable` if required RNA information is missing. Preserve leading zeros when importing. |
| `recovery_pattern_label` | Text | Human-readable joined feature names, `none`, or `unavailable`. |

Missing RNA assessments withhold the corresponding scores and combined recovery
metrics. Missing completeness/contamination withholds the integrity/composite
interpretation. A true absence after a successful assessment remains zero.

### Atlas interpretation of GUNC

| Column | Meaning |
| --- | --- |
| `gunc_assessment` | `unscored` if RRS/CSS is missing; else `low_reference_uncertain` if RRS <0.3; else `credible_chimeric_signal` if CSS >0.45; else `likely_not_chimeric`. |
| `gunc_assessment_label` | Display label for `gunc_assessment`. |
| `gunc_strict_assessment` | Missing RRS/CSS → `unscored`; RRS <0.5 → `low_reference_uncertain`; otherwise CSS ≥0.85 and rank above genus → `credible_chimeric_signal`; otherwise `likely_not_chimeric`. This is an additional Genome QC rule, not GUNC's default pass flag. |
| `gunc_strict_assessment_label` | Display label for the stricter interpretation. |
| `gunc_low_reference_uncertain`, `gunc_credible_chimeric_signal`, `gunc_likely_not_chimeric`, `gunc_unscored` | 0/1 indicators of the corresponding `gunc_assessment` category. |
| `gunc_strict_low_reference_uncertain`, `gunc_strict_credible_chimeric_signal`, `gunc_strict_likely_not_chimeric`, `gunc_strict_unscored` | 0/1 indicators of the corresponding `gunc_strict_assessment` category. |
| `gunc_above_genus_conflict` | 0/1: reported rank is kingdom, phylum, class, order or family. This flag alone does not establish contamination. |
| `gunc_strict_chimera` | 0/1: RRS ≥0.5, CSS ≥0.85 and rank above genus; missing comparisons do not pass. |

The word “strict” denotes stronger evidence required to call a chimeric signal;
it does not mean a more aggressive exclusion filter. None of these fields is an
automatic final-selection requirement.

## Atlas summaries

`<prefix>_summary.tsv` is one row for the cohort. `<prefix>_group_summary.tsv`
has the same measurement summaries per selected grouping column (default
`Phylum`). Means and medians exclude missing numerical values, so denominators
can differ between measurements; `n_genomes` is the group size, not a per-metric
nonmissing count.

| Columns | Meaning / units |
| --- | --- |
| `n_genomes` | Count of rows/genomes in the summarized cohort or group. |
| `median_qscore`, `mean_qscore` | Median and arithmetic mean Qscore. |
| `median_integrity`, `mean_integrity` | Summary of `integrity_score` (0–1). |
| `median_recoverability`, `mean_recoverability` | Summary of `recoverability_score` (0–1). |
| `median_mimag_quality_index`, `mean_mimag_quality_index` | Summary of the composite index (0–1). |
| `median_completeness`, `mean_completeness` | Completeness summaries (%). |
| `median_contamination`, `mean_contamination` | Contamination summaries (%). |
| `median_16S`, `mean_16S`, `median_23S`, `mean_23S`, `median_5S`, `mean_5S` | Summaries of the capped rRNA scores, not raw copy-number counts. |
| `median_trna_ge_18`, `mean_trna_ge_18` | Summaries of the tRNA hallmark indicator. Its mean is the fraction meeting the criterion among scored genomes. |
| `median_recovered_feature_count`, `mean_recovered_feature_count` | Summary of the number of recovered hallmarks (0–4). |
| `high_quality_n`, `medium_quality_n`, `low_quality_n`, `failed_quality_n` | Counts in the corresponding `mimag_tier`; unavailable tiers are not a “failed” quality call. |
| `Phylum` (or configured grouping header) | Group being summarized; values come from taxonomy or supplied metadata. |

`<prefix>_feature_recovery.tsv` has `feature` (hallmark/score name) and
`fraction_present` (mean binary hallmark indicator across scored genomes).
For rRNAs this is presence of any positive weighted score, not mean fractional
rRNA recovery; for tRNA it is the fraction meeting the ≥18-isotype criterion.

## Cohort ANI and phylogeny

### `ani/ani_pairs.tsv`

One row per pair with at least one reported directed FastANI result. An absent
pair is unreported, not zero ANI; self comparisons are excluded.

| Column | Units | Meaning |
| --- | --- | --- |
| `genome_a`, `genome_b` | IDs | Original input identifiers of the pair. |
| `ani_min` | Percent | Smaller of the two directed ANI estimates; blank without reciprocal results. |
| `af_a`, `af_b` | Fraction | Mapped query fragments / total query fragments for A→B and B→A, respectively. |
| `af_min` | Fraction | Smaller of the two directional AF values; blank without reciprocal results. |
| `af_larger`, `af_smaller` | Fraction | AF when the larger or smaller assembly is the query, based on total assembly length. |
| `passes` | Boolean | Reciprocal results, minimum ANI ≥configured threshold, and either larger-assembly AF ≥configured threshold or enabled containment rescue. Defaults: ANI 98%, AF 0.51; rescue: smaller AF ≥0.90 and larger AF ≥0.10. |
| `match_type` | Enum | `standard`, `containment`, or `unmatched`. A passing standard match takes precedence over rescue. |
| `status` | Enum | `reciprocal` or `one_direction_only`. |

The standard AF criterion is **larger-assembly AF**, not necessarily `af_min`.
See [phylogeny](phylogeny.md) for the biological limitations of containment rescue.

### Other cohort files

| File / columns | Meaning |
| --- | --- |
| `ani_clusters.tsv`: `Genome_Id`, `ani_cluster`, `cluster_size` | Connected-component membership and number of genomes in that component. `ANI_0001` etc. are deterministic within the cohort, not permanent species IDs. Connected groups need not satisfy all-pairs thresholds. |
| `ani_status.tsv`: `Genome_Id`, `status`, `reason` | `compared` or `not_compared`. Reasons include `no_contig_at_least_3000_bp` and `fewer_than_two_eligible_genomes`. A singleton cluster alone does not prove distinctness. |
| `phylogeny_status.tsv`: `Genome_Id`, `domain`, `status`, `reason` | Per-genome tree participation (`inferred` or `not_inferred`) and reason, including `fewer_than_three_usable_aligned_genomes` or `no_usable_GTDB_marker_alignment`. `domain` identifies `bac120` or `ar53`; inspect status rather than assuming all genomes occur in a tree. |
| `<domain>.tip_ids.tsv`: `tip`, `Genome_Id` | Safe tree tip label mapped to original genome ID. |
| `<domain>_annotated_tree.tips.tsv` | Master columns plus `ani_cluster`, one row per included genome. The same master definitions apply. |

`ani/fastani.raw.tsv` is a **headerless** upstream file: five positional fields
are query FASTA, reference FASTA, ANI (%), mapped fragments and total query
fragments. Read it with `header=None`; the first row is data. The final named
ANI tables are the recommended explorer/analysis inputs.

## Intermediate and upstream tables

These support auditing and standalone stages. The final master/atlas tables are
usually easier to join. Native upstream fields are preserved rather than renamed;
the links below provide the detailed upstream definitions for version-specific
outputs. Additional raw diagnostic files can vary with the tool version/domain.

### SeqKit, CheckM and variant selection

`seqkit_stats.tsv` and individual `*.seqkit.tsv` describe each variant.
`qscore_all.tsv`, `dedupe_all.tsv`, and `dedupe_assessed.tsv` reuse fields from
SeqKit, CheckM, GTDB and the rRNA comparison, plus the identifiers/Qscore above.

| Column | Meaning / units |
| --- | --- |
| `file` | Staged FASTA filename/path; may point to removed temporary work after compaction. Use the input inventory for original source paths. |
| `format`, `type` | SeqKit file format and sequence alphabet, normally FASTA and DNA. |
| `Q1`, `Q2`, `Q3` | Sequence-length quartiles in bp; `Q2` is median contig length. |
| `sum_gap` | Total gap characters counted by SeqKit. |
| `N50_num` | Number of longest sequences needed to reach N50's cumulative-length threshold (L50 count). |
| `Q20(%)`, `Q30(%)`, `AvgQual` | Read-quality summaries when the source supports quality values; not meaningful quality scores for FASTA assemblies and normally unavailable there. |
| `GC(%)` | SeqKit GC percentage. |
| `sum_n` | Number of ambiguous N bases counted by SeqKit. |
| `Marker lineage` | CheckM lineage marker set chosen for the genome. |
| `# genomes` | Reference genomes supporting that CheckM marker lineage. |
| `# markers`, `# marker sets` | Numbers of individual marker genes and marker sets used by CheckM. |
| `0`, `1`, `2`, `3`, `4`, `5+` | Counts of marker genes observed in the indicated copy-number category (zero, once, twice, etc.; last is at least five). |
| `requested_id`, `matched_fasta`, `status` | `cp_dedupe_audit.tsv`: requested representative, source FASTA found, and copy/match outcome. |

`num_seqs`, `sum_len`, `min_len`, `avg_len`, `max_len`, `N50`, the three CheckM
quality fields, `qscore` and identifiers have the same definitions as in the
master. See [SeqKit stats](https://bioinf.shenwei.me/seqkit/usage/#stats) and
[CheckM output](https://github.com/Ecogenomics/CheckM/wiki/Reported-Statistics).

### Parsed GFF and RNA tables

`barrnap_gff_table.tsv` and `trnascan_gff_table.tsv` retain GFF fields and add
parsed attributes. The row unit is one feature, not one genome.

| Column | Meaning / units |
| --- | --- |
| `seqid` | Source contig identifier. |
| `source` | Feature-producing software/source from GFF. |
| `type` | GFF feature type. |
| `start`, `end` | GFF 1-based inclusive coordinates on the contig. |
| `score` | Predictor-specific GFF score; `.` if absent. Do not compare scores across different predictors. |
| `strand` | `+`, `-`, or unspecified strand notation. |
| `phase` | GFF phase; normally `.` for RNA features. |
| `attributes` | Original semicolon-separated GFF attributes. |
| `genome_id` | Variant identifier assigned from the prediction filename, corresponding to `Bin Id`. |
| `name`, `product` | Parsed barrnap feature name/subunit and product description. |
| `prod_mod` | Per-rRNA weight: reported “aligned only … percent” /100 for a partial prediction; otherwise 1. |
| `isotype` | Parsed tRNA isotype label from GFF. |
| `subunit_count` | In `barrnap_subunit_counts.tsv`, sum of `prod_mod` per `genome_id` and `name`; may be fractional. |
| `trna_total`, `trna_unique` | In `trnascan_trna_counts.tsv`, total features and distinct isotype labels per `genome_id`. |

### rRNA BLAST tables

`barrnap_blast_raw.tsv` contains matching-subunit hits; the pass table contains
per-variant minima of identity and bit score. Successful no-hit rows in the pass
table use **−1** in `pident` and `bitscore` as sentinels, not measured identities.

| Column | Meaning / units |
| --- | --- |
| `qaccver`, `saccver` | Query and subject RNA sequence identifiers. |
| `pident` | Alignment identity (%); per-variant minimum in the pass table. |
| `length` | Alignment length in nucleotide positions, including aligned gaps. |
| `mismatch`, `gapopen` | Alignment mismatch and gap-opening counts. |
| `qstart`, `qend` | 1-based query alignment endpoints. |
| `sstart`, `send` | 1-based subject endpoints; reverse-strand hits can have descending coordinates. |
| `evalue` | BLAST expected chance-hit count at this score under its search model. |
| `bitscore` | BLAST bit score; per-variant minimum in the pass table. |
| `SSU`, `sSSU` | Subunit-name prefixes parsed from query and subject IDs; matching names are retained. Despite the column names, this comparison can include different rRNA subunit types. |
| `Bin Id`, `pass_BARRNAP` | Variant and consistency result as defined above. |

### GTDB-Tk classification and supporting tables

The merged and individual `<prefix>.bac120.summary.tsv` and
`<prefix>.ar53.summary.tsv` share a schema. A blank result is not a zero ANI.
The following descriptions concern GTDB reference classification, **not** the
separate cohort FastANI thresholds.

| Column | Meaning / units |
| --- | --- |
| `user_genome` | GTDB input variant ID. |
| `classification` | Semicolon-separated GTDB rank string, including rank prefixes. |
| `closest_genome_reference` | Reference accession associated with closest-genome ANI classification. |
| `closest_genome_reference_radius` | Reference species ANI boundary (%). |
| `closest_genome_taxonomy` | Taxonomy of that reference. |
| `closest_genome_ani`, `closest_genome_af` | Query/reference ANI (%) and alignment fraction. |
| `closest_placement_reference` | Reference associated with tree placement. |
| `closest_placement_radius` | Placement reference's species ANI boundary (%), where reported. |
| `closest_placement_taxonomy` | Placement reference taxonomy. |
| `closest_placement_ani`, `closest_placement_af` | Placement-reference ANI (%) and alignment fraction. |
| `pplacer_taxonomy` | Tree-derived taxonomy before final classification reconciliation. |
| `classification_method` | Rule/evidence used for the final taxonomy. |
| `note`, `warnings` | Upstream classification diagnostics. |
| `other_related_references(genome_id,species_name,radius,ANI,AF)` | Related-reference tuples; parse as a structured list, not one scalar measurement. |
| `msa_percent` | Aligned amino-acid representation (%). |
| `translation_table` | NCBI genetic-code number used for gene prediction. |
| `red_value` | Relative evolutionary divergence associated with placement. |

See the [GTDB-Tk summary schema](https://ecogenomics.github.io/GTDBTk/files/summary.tsv.html).
Interpret classifications against the database release recorded with the run.

Additional current GTDB diagnostic tables (see also the
[marker summary definitions](https://ecogenomics.github.io/GTDBTk/files/markers_summary.tsv.html)):

| File / columns | Meaning |
| --- | --- |
| `*.ani_summary.tsv`: `user_genome`, `reference_genome`, `skani_ani`, `skani_af`, `reference_taxonomy` | ANI-screen query/reference match, ANI (%), AF (fraction), and reference taxonomy. Related-reference tuples use the summary definition above. |
| `*.tree.mapping.tsv`: `user_genome`, `is_ani_classification`, `class_tree_mapped`, `classification_rule` | Query ID, whether ANI supplied classification, class-level tree chosen, and the internal classification rule. |
| `*.markers_summary.tsv`: `name` | Query variant ID; this use of `name` differs from the RNA subunit-name column. |
| `number_unique_genes`, `number_multiple_genes`, `number_multiple_unique_genes`, `number_missing_genes` | Marker categories: one copy, multiple differing copies, multiple identical copies, and missing markers, respectively. |
| `list_unique_genes`, `list_multiple_genes`, `list_multiple_unique_genes`, `list_missing_genes` | Marker identifiers belonging to the corresponding category. |
| `*.translation_table_summary.tsv` | Headerless: query ID then genetic-code number. |
| `*.filtered.tsv`, `*.failed_genomes.tsv` | Diagnostic ID/reason records; can be empty and may be headerless. Read the native format rather than assuming the first row contains headers. |

GTDB also retains version-dependent gene-call and marker files. Consult the
[GTDB file reference](https://ecogenomics.github.io/GTDBTk/files/index.html) for
native diagnostic formats beyond the tabular products described here.

### Reference readiness receipts

`references/*_reference_ready.tsv` contains `reference` (database identifier),
`path` (resolved installed reference location), and `status` (readiness result).
A readiness receipt is not a per-genome biological assessment. Database version,
archive checksum and installation provenance are described in [databases](databases.md).

## Statistical and optional comparison tables

The default atlas can write an empty `<prefix>_group_stats.tsv` when there is no
eligible comparison design. Empty means no supported test was produced, not that
all comparisons were nonsignificant. Optional `atlas --compare-column` and
`compare` runs add study-specific contrasts and component/selection summaries.
See [interpretation](interpretation.md#comparisons-between-labelled-sets) for
model assumptions and adjustment families.

### Mean-comparison and significance fields

The following fields are shared by group/method comparison tables. Some appear
only when a test can be constructed. A missing p-value must be interpreted with
`status`, not converted to 1 or 0.

| Column | Meaning / units |
| --- | --- |
| `test` | Statistical procedure used; identifies the method rather than its result. |
| `metric`, `metric_label` | Machine-readable response name and display label. Effect sizes retain that response's units. |
| `group_column`, `sample_column` | Input metadata columns identifying methods/groups and samples. |
| `analysis_scope` | Analysis family and global/pairwise contrast scope; also used to define correction families. |
| `sample` | Sample scope; `all_samples` identifies the pooled sample-adjusted model. |
| `samples`, `n_samples` | Semicolon-separated contributing sample labels and their count. |
| `groups`, `n_groups` | Included category labels and their count. |
| `group_a`, `group_b` | Categories compared in a pairwise contrast; blank for an omnibus/global test. |
| `n_genomes`, `n_a`, `n_b` | Total analyzed observations and per-category counts after required-value filtering. |
| `n_clusters` | Number of covariance dependency/anchor groups; not ANI cluster count. |
| `min_category_clusters` | Minimum dependency-group count across categories involved in the contrast. |
| `n_unresolved` | Number of observations with unresolved singleton/xPG anchor assignments. |
| `blocking_unit`, `aggregation` | Dependency unit and aggregation rule; the current mean model uses anchor groups and no pre-aggregation. |
| `estimand`, `model` | Target contrast and model description. Current mean comparisons estimate sample-adjusted differences using additive method and sample terms. |
| `covariance`, `working_correlation` | GEE covariance and working-correlation choices; working independence does not assert biological independence. |
| `group_a_mean`, `group_b_mean` | Observed raw arithmetic means, not sample-adjusted estimates. |
| `group_a_median`, `group_b_median` | Observed raw medians. |
| `group_a_adjusted_mean`, `group_b_adjusted_mean` | Model means standardized to the same observed sample distribution. |
| `mean_difference_a_minus_b` | Adjusted A minus B mean difference. Positive values favor a larger response in A; “larger” is not necessarily better quality for every metric. |
| `standard_error` | Anchor-robust standard error of the pairwise mean difference, in response units. |
| `ci_low`, `ci_high` | Unadjusted robust 95% confidence limits of that difference; not simultaneous/BH-adjusted intervals. |
| `statistic`, `df` | Wald statistic and degrees of freedom for the reported contrast. |
| `pvalue` | Unadjusted primary-test p-value. |
| `qvalue_bh` | Benjamini–Hochberg adjusted primary p-value within the documented analysis family. |
| `standard_error_independent`, `pvalue_independent`, `qvalue_independent_bh` | Sensitivity analysis using singleton covariance groups with the same mean design; not a second independent biological experiment. |
| `significant_p05`, `significant_q05`, `significant_independent_q05` | Whether the corresponding p/q-value is strictly below 0.05. Missing values do not pass. |
| `significance_stars` | Primary BH annotation: `*` <0.05, `**` <0.01, `***` <0.001; blank otherwise or if unavailable. |
| `status` | `ok` or an explicit reason testing was not supported, such as `insufficient_clusters`, `insufficient_category_clusters`, `unresolved_xpg_anchor`, `confounded_method_sample_design`, `constant_outcome`, `not_converged`, `nonfinite_covariance`, `singular_contrast_covariance`, or `fit_failed: …`. |

Dependency audit tables add `dependency_genome_id` (resolved row identity),
`dependency_group` (stable hash of the related sample/anchor members), and
`dependency_status` (`source_anchor`, `derived_anchor`, `unresolved_singleton`,
or `unresolved_xpg_anchor`). These group covariance dependencies; they do not
replace the genome IDs or establish species assignments.

Comparison tables also retain user-selected category/sample fields and copies
of master/atlas metrics, sometimes with category prefixes. Those measurements
keep the definitions above. Paired/component differences use the stated A-minus-B
orientation and component membership rather than independent genome replicates.
Do not mix these optional comparison outputs with the default cohort
`ani_pairs.tsv` schema: they answer different questions and can use different
thresholds. The [full standalone CLI reference](cli-reference.md#atlas) lists the
relevant options.

## Provenance and non-tabular outputs

| File / field family | Meaning |
| --- | --- |
| `inputs.resolved.json`: `id`, `fasta`, `sha256` | Original input ID, resolved FASTA path and file checksum; additional keys are supplied manifest metadata. |
| `logs/<run-id>/run.json`: `invocation`, `command`, `database_command`, `params`, `resources` | Parsed invocation, exact workflow commands, resolved parameters and scheduling requests. |
| `run.json`: `version`, `started_utc`, `status`, `assessment_failures` | Controller version, run identifier/time, completion state, and number of unresolved assessment/inventory records when counted. Do not interpret that count as a count of unique failed genomes. |
| `run.json`: `controller_limits` | Detected Slurm-controller resource constraints, when recorded. These are distinct from worker requests. |
| `environment-packages.json` | Installed tool-environment package name/version/build/channel/checksum records. A missing file after a provenance warning requires checking the logs. |
| `compact-results.json`: `signature`, `products`, `cleanup_complete` | Input/settings/implementation signature, relative retained-product paths and SHA-256 values, and whether private cleanup finished. A `complete` run receipt can precede completion of cleanup. |
| `ani/provenance.json` | Actual ANI/AF rules, containment settings, clustering semantics and cohort size. |
| Exported selection JSON | Table source, filters, search, selected columns/cohort and, for selection receipts, matching genome IDs and input-table checksum. It records a selection; it does not recalculate figures. |
| `trace.tsv` | Native Nextflow task execution/resource fields, not genome measurements; see below. |

Typical Nextflow trace columns are `task_id` (task identifier), `hash` (work/cache
key), `native_id` (local PID or scheduler job ID), `name`, `status`, `exit`
(exit code), `submit` (submission time), `duration` (task duration), `realtime`
(execution runtime), `%cpu`, `peak_rss`, `peak_vmem`, `rchar`, and `wchar`.
CPU percentage can exceed 100 for parallel tasks; RSS is resident memory, VMEM
is virtual address space, and `rchar`/`wchar` count process I/O characters rather
than physical disk traffic. Time/resource formatting and the exact fields depend
on Nextflow. Use [Nextflow's trace definitions](https://www.nextflow.io/docs/latest/tracing.html#trace-report)
when deriving benchmark statistics; summing overlapping task durations is not
workflow wall time.
