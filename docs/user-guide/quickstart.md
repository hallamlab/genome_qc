# Quickstart

[Install Genome QC](installation.md), then put your genome FASTAs in an input
directory. Each file represents one genome. Compressed FASTAs are accepted.
Use a server suitable for the [reference and memory requirements](resources.md).

## Run from FASTAs to the atlas

The following assumes your files are in `~/genomes`:

```bash
conda activate genome_qc
genome_qc run \
  -i ~/genomes \
  -o ~/genome-qc-output \
  -d ~/genome-qc-databases \
  --threads 8 --max_cpus 32
```

The command prepares missing databases and tool environments, assesses genomes,
and builds the final tables, atlas and explorer. Existing complete databases are
reused. It retains all contigs and applies no quality acceptance thresholds by
default. A low-quality genome still receives every applicable assessment.

`--threads 8` permits eight CPUs per parallel tool; `--max_cpus 32` budgets 32 CPUs
across local jobs. Memory reservations can further limit concurrency. Balanced
batches are automatic, with a maximum of 1,000 genomes per batch.

For explicit genome IDs, replace `-i ~/genomes` with
`--manifest ~/genomes.tsv`; see [input organization](inputs.md).

## Open the results

```bash
genome_qc report -o ~/genome-qc-output --serve --port 8765
```

Open `http://127.0.0.1:8765` on the machine running the command. For an SSH session,
open a second terminal **on your own computer**:

```bash
ssh -N -L 8765:127.0.0.1:8765 YOUR_USER@YOUR_SERVER
```

Then visit the same local URL in your browser. The report server listens on the
loopback interface. Start with **Summary & atlas**, inspect subsets in **Explore
tables**, and save documented subsets from **Select genomes**.

## Keep a compact completed result

For a fresh compact run:

```bash
genome_qc run \
  -i ~/genomes \
  -o ~/genome-qc-output-compact \
  -d ~/genome-qc-databases \
  --threads 8 --max_cpus 32 --compact_results
```

Cleanup happens only after the full cohort, atlas and report complete without
outstanding assessment failures. Tables, figures, selected genome sequences,
provenance and diagnostic logs remain. Original inputs and databases are never
removed. Read [compact mode](recovery.md) before relying on it for scratch quotas.
