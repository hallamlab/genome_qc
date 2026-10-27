# Check your setup and validate results

## Quick setup check (recommended)

After [installation](installation.md):

```bash
conda activate genome_qc
genome_qc --version
genome_qc check
genome_qc run --help
```

These checks do not download databases or validate biological predictions.
`check` verifies the availability of Nextflow, Java and Mamba on `PATH`.
This is the recommended initial installation check; the full analysis below
is optional and is not required just to verify the controller setup.

## Optional full biological validation

**Allow hours, not minutes.** A few genomes still require large reference trees
and indices; disk-backed GTDB placement can dominate runtime. This validation
is for checking the complete scientific workflow on suitable hardware, not a
quick installation test.

Put a few bacterial or archaeal genome FASTAs you are permitted to use in
`~/genome-qc-test-inputs`. Use unique filenames. A small genome set reduces task
work, but loading the reference trees still requires a suitable server.
The repository does not currently provide a public real-genome test bundle.
The local 300-genome regression set used during development is not packaged.

```bash
genome_qc run -i ~/genome-qc-test-inputs \
  -o ~/genome-qc-test-output -d ~/genome-qc-databases \
  --threads 8 --max_cpus 32
genome_qc report -o ~/genome-qc-test-output --serve --port 8765
```

Review the input inventory, master table, `Assessment_status.tsv`, atlas and
explorer. Every supplied genome should be accounted for. Missing measurements
must have an interpretable status; report creation alone does not establish that
all tools succeeded. Repeat the exact `run` command with `--resume` to check
cache reuse. For a fresh compact-mode test, use a **new** output directory and
add `--compact_results`.

## Check the results

Use this checklist before starting a larger analysis:

1. **Inputs accounted for:** compare the inventory's genome IDs with your input
   filenames or manifest. Unexpected IDs often indicate extra FASTAs in a nested directory.
2. **Assessments complete:** inspect `Assessment_status.tsv` and the explorer's
   incomplete-assessment count. A successful program exit and a biologically
   acceptable genome are different questions.
3. **Measurements present:** inspect assembly lengths, completeness/contamination,
   taxonomy and RNA counts for several genomes. A genuine zero RNA count is
   different from an unavailable measurement.
4. **ANI and tree outputs present:** check the ANI and phylogeny status tables.
   A marker tree requires at least three usable genomes in that domain; a
   smaller test may legitimately have no tree. See [phylogeny](phylogeny.md).
5. **Explorer usable:** open a figure, filter a table and export a small subset.
   Confirm that the exported IDs match your intended selection.
6. **Resume behaves as expected:** for an unmodified noncompact run, repeat the
   command with `--resume` and look for cached tasks. A fully completed compact
   run instead verifies and reuses its retained final results.

No quality thresholds are enabled by default, so it is normal for all assessed
genomes to appear in `Selected_genome_QC.tsv`, including poor-quality genomes.
Use [final selection](selection.md) when you are ready to define acceptance criteria.

## Developer tests

The repository has unit tests plus opt-in Nextflow integration tests with synthetic
genomes and simulated tool responses. They exercise batching, failure recovery,
merging, reporting, compact cleanup and resume. They are not a biological demo
and do not establish accuracy of CheckM, GTDB-Tk or GUNC. See
[development](development.md) for their requirements.
