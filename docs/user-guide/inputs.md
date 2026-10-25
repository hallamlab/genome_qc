# Inputs and identifiers

## Genome directories

Supply bacterial or archaeal genome FASTAs with `-i`. The controller searches
nested directories for `.fa`, `.fna`, `.fasta`, and gzip-compressed equivalents.
Each file is one genome, not one contig. Directory mode uses the filename stem
as the genome ID; stems must be unique across the input tree.

The complete workflow is intended for prokaryotic genomes. The standalone RNA
commands expose model choices, but this does not make the complete CheckM and
GTDB-Tk workflow a eukaryotic genome assessment pipeline.

## Explicit manifests

A tab-separated manifest requires `id` and `fasta` columns:

```text
id	fasta
isolate_01	genomes/isolate_01.fna.gz
MAG_02	genomes/MAG_02.fasta
SAG_03	genomes/SAG_03.fa
```

Paths resolve relative to the manifest; absolute paths are also accepted. IDs
must start with a letter or number and contain only letters, numbers, periods,
underscores or hyphens. Missing files and duplicate IDs stop input validation.
The controller records SHA-256 checksums and resolved paths in
`inputs.resolved.json`. Reusing an output with changed inputs is rejected.
Additional manifest columns are retained as input provenance; a full run does
not automatically interpret them as an experimental design for atlas comparisons.

## Contig-length variants

All contigs are retained by default. To assess multiple minimum-length variants:

```bash
genome_qc run --manifest ~/genomes.tsv \
  -o ~/genome-qc-variants -d ~/genome-qc-databases \
  --filter_size '1000 2000' --threads 8 --max_cpus 32
```

Each nonempty variant receives sequence statistics, CheckM, taxonomy and rRNA
checks. One variant per original genome is ranked by Qscore, then N50, then
assembly length and proceeds to tRNA/GUNC assessment. This is selection among
variants of the same input genome, not ANI dereplication between different genomes.
Genomes with no usable variant remain visible in the input inventory.

## Comparing sets

`genome_qc compare` uses a different manifest: `sample`, `category`, and
`run_directory`, with an optional `bin_id_token_index`. Its rows refer to completed
Genome QC runs, not FASTA files. See [standalone analyses](commands.md).
