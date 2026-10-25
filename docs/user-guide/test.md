# Test the installation

## Check the controller

After [installation](installation.md):

```bash
conda activate genome_qc
genome_qc --version
genome_qc check
genome_qc run --help
```

These checks do not download databases or validate biological predictions.

## Run a small real-genome test

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

## Developer tests

The repository has unit tests plus opt-in Nextflow integration tests with synthetic
genomes and simulated tool responses. They exercise batching, failure recovery,
merging, reporting, compact cleanup and resume. They are not a biological demo
and do not establish accuracy of CheckM, GTDB-Tk or GUNC. See
[development](development.md) for their requirements.
