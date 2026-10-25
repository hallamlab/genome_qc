# ANI groups and marker phylogeny

A complete `genome_qc run` includes cohort-wide FastANI comparisons and annotated
marker trees. They use every assessed representative, not only the final selected
subset, and do not require category labels. The results appear in **Summary &
atlas** in the explorer, with searchable supporting tables in **Explore tables**.

## Default matching rules

The default ANI cutoff is **98%** and the normal alignment-fraction (AF) cutoff is
**0.51**. FastANI reports matched fragments and total query fragments; their ratio
is the query's AF. Both directions must be reported, and the lower ANI must meet
the cutoff. Assembly length identifies the larger and smaller genomes.

A pair passes either of these rules:

- **Standard:** the larger assembly has AF ≥0.51.
- **Containment:** the smaller assembly has AF ≥0.90 and the larger assembly has
  AF ≥0.10. This can recover matches between incomplete and more complete genomes.

The pair table records both AF values and labels accepted pairs `standard` or
`containment`. Missing FastANI results are unreported comparisons, not zero ANI. Genomes
without a contig long enough for FastANI's 3,000 bp fragments are explicitly
recorded as not compared in `ani_status.tsv`, while remaining in the master table.
Containment supports similarity of recovered sequence; it does not establish
complete-genome equivalence. Connected components define ANI groups: a chain of
accepted pairs can join genomes that do not themselves pass the pairwise cutoff.
Singletons can reflect insufficient sequence or absent matches, not necessarily
biological uniqueness. Group IDs are deterministic within a fixed cohort.

```bash
genome_qc run -i ~/genomes -o ~/genome-qc-output \
  -d ~/genome-qc-databases --threads 8 --max_cpus 32 \
  --ani_threshold 98 --ani_af_threshold 0.51
```

Use `--no_ani_containment` to require the normal AF rule for every accepted pair.
`--skip_ani` and `--skip_phylogeny` independently disable the new stages. These
scientific settings require a new output directory if changed. ANI groups are
annotations; they do not discard input genomes or force one representative per group.

## Trees and annotations

The workflow preserves the small GTDB masked user alignments before task-level
compact cleanup. It combines matching alignment columns across batches, selects
the same contig-length representative as the master table, and validates sequence
lengths and duplicate IDs. No additional GTDB database download or marker search
is needed. Genomes with unresolved species names remain eligible.

Bacterial **bac120** and archaeal **ar53** alignments produce separate trees with
FastTree 2.1.11 using LG, a fixed seed of 42, and gamma branch-length rescaling.
FastTree performs approximate maximum-likelihood inference; its local SH-like
supports are not bootstrap probabilities. Filled nodes mark local support ≥0.95.
The root is an arbitrary display root, not an inferred ancestral root. Trees with
fewer than three usable aligned genomes are not inferred; missing alignments and
small-domain cohorts are recorded in `phylogeny_status.tsv`.

Figures label genomes and GTDB species, color labels by phylum, and include
completeness, contamination and ANI groups. The matching tip-metadata TSV retains
all master-table fields. PNG previews, vector PDF/SVG figures, Newick trees,
alignments, command records and inference logs are retained in compact results.
These are marker-based trees, not trees derived from ANI distances. Changing ANI
thresholds changes group annotations, not the marker alignment or tree method.

FastANI and FastTree use the task CPU allocation (`--threads`, bounded by
`--max_cpus` locally); the atlas memory override also applies to these tasks.
The FastTreeMP executable is preferred when available. Nextflow schedules the
stages on the selected local/Slurm executor, with the same scratch and resume
behavior as the rest of the workflow. All-versus-all ANI work grows with cohort
size; these stages may add substantial runtime for large genome collections.

## References

FastANI: Jain et al. (2018), *High throughput ANI analysis of 90K prokaryotic genomes
reveals clear species boundaries*. [DOI: 10.1038/s41467-018-07641-9](https://doi.org/10.1038/s41467-018-07641-9).
[Output format and alignment fraction](https://github.com/ParBLiSS/FastANI).

FastTree: Price, Dehal and Arkin (2010), *FastTree 2—Approximately Maximum-Likelihood
Trees for Large Alignments*. [DOI: 10.1371/journal.pone.0009490](https://doi.org/10.1371/journal.pone.0009490).
[Software and methods](https://www.microbesonline.org/fasttree/).
See also the [GTDB-Tk and GTDB references](citations.md).
