# Inputs and identifiers

## Genome directories

Supply bacterial or archaeal genome FASTAs with `-i`. The controller searches
nested directories for `.fa`, `.fna`, `.fasta`, and gzip-compressed equivalents.
Each file is one genome, not one contig. Directory mode uses the filename stem
as the genome ID; stems must be unique across the input tree.

The complete workflow is intended for prokaryotic genomes. The standalone RNA
commands expose model choices, but this does not make the complete CheckM and
GTDB-Tk workflow a eukaryotic genome assessment pipeline.

### Avoid common input mistakes

- Supply nucleotide assemblies, not protein FASTAs or sequencing reads.
- Use a directory containing only the genomes intended for this analysis.
  Discovery is recursive; files in nested directories are included too.
- `sample.fna` and `sample.fasta.gz` both become ID `sample` and therefore conflict.
  Rename them or assign distinct manifest IDs.
- Use stable IDs without spaces. The same IDs connect taxonomy, quality metrics,
  RNA predictions, ANI groups and tree tips throughout the results.
- If input paths contain spaces, quote the paths in shell commands. A manifest
  stores the path itself without shell quotation marks.

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

### Create a small manifest in a terminal

This example creates a real tab-separated file. Adjust the FASTA filenames to
match your files first:

```bash
printf 'id\tfasta\nisolate_01\tgenomes/isolate_01.fna\nMAG_02\tgenomes/MAG_02.fasta.gz\n' > ~/genomes.tsv

genome_qc run --manifest ~/genomes.tsv \
  -o ~/genome-qc-output -d ~/genome-qc-databases \
  --threads 8 --max_cpus 32
```

Because the manifest is in your home directory, `genomes/isolate_01.fna` resolves
to `~/genomes/isolate_01.fna`. If you create the table in a spreadsheet, export it
as tab-separated text, not CSV. Supply either `-i` or `--manifest`, not both.

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

Directory discovery is recursive by default. Add `--no_recursive` to restrict `-i`
to files directly inside that directory. The CLI prints counts for each discovered
parent directory. Use a manifest when excluding particular genomes or subfolders.
