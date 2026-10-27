# Quickstart

## Check your setup first

[Install Genome QC](installation.md), then verify the controller in your terminal:

```bash
conda activate genome_qc
genome_qc --version
genome_qc check
genome_qc run --help
```

These commands do not download reference databases or run genome assessments.
`check` confirms that Nextflow, Java and Mamba are available on `PATH`; it does
not test biological tools or database contents. You do not need to run a
three-genome analysis to check the installation. Even a few genomes can take
hours in GTDB reference-tree placement, particularly with disk-backed scratch.

Once the setup check passes, use your own genome FASTAs for the analysis below.
Full biological validation is optional and described in the [test guide](test.md).

## Before you begin

You need an installed copy of Genome QC, bacterial or archaeal genome FASTAs,
and a Linux server with enough memory and storage for the reference databases.
These are assembled genomes, SAGs or MAGs, not FASTQ reads. You do not need
MetaPathways, Pathway Tools, sample metadata or a pre-existing QC table for a full run.

The examples use three separate locations:

| Location | What belongs there | Reuse between runs? |
| --- | --- | --- |
| `~/genomes` | Your input genome FASTAs, one genome per file | Yes; keep them unchanged during a run |
| `~/genome-qc-databases` | Large shared reference databases | Yes; do not make a new copy for each analysis |
| `~/genome-qc-output` | Results, logs and private working files for this analysis | Use a new directory for a different input set |

`~` means your home directory. Replace these locations with your own paths;
on a cluster, shared project or scratch paths are usually more appropriate.
Keep the output outside the input directory: input discovery searches subdirectories
and could otherwise pick up FASTAs produced by a previous analysis.

For example, an input directory could contain:

```text
genomes/
├── isolate_01.fna
├── MAG_02.fasta.gz
└── SAG_03.fa
```

Each file may contain many contigs belonging to that genome. Do not split each
contig into a separate input file. See [inputs](inputs.md) for explicit IDs and manifests.

Activate the software environment in each new terminal and check it:

```bash
conda activate genome_qc
genome_qc check
```

Choose CPU limits that fit your server. The examples below assume 32 available
CPUs; they are not a requirement. [Resource settings](resources.md) explain how
to adapt them. Small-input analyses still load the same large reference databases
as larger analyses.

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
and builds the final tables, ANI groups, marker phylogenies, atlas and explorer.
[ANI and phylogeny](phylogeny.md) run by default at 98% ANI and 51% AF, with
labelled containment rescue for incomplete assemblies. Existing complete databases are
reused. It retains all contigs and applies no quality acceptance thresholds by
default. A low-quality genome still receives every applicable assessment.

`--threads 8` permits eight CPUs per parallel tool; `--max_cpus 32` budgets 32 CPUs
across local jobs. Memory reservations can further limit concurrency. Balanced
batches are automatic, with a maximum of 1,000 genomes per batch.

For explicit genome IDs, replace `-i ~/genomes` with
`--manifest ~/genomes.tsv`; see [input organization](inputs.md).

## What to expect while it runs

The first run can take substantially longer than later runs:

1. Genome QC validates input IDs and paths, records checksums, and stages FASTAs.
2. It checks the reference root and prepares missing databases. Downloads and
   extraction need network access and substantial free disk space.
3. Nextflow creates the tool environments and schedules the assessment tasks.
   A queued task may be waiting for a dependency, CPU reservation or memory reservation.
4. The workflow combines assessments, calculates ANI groups and marker trees,
   builds the atlas, and indexes the explorer.

Keep the controller terminal/session alive until it finishes. On a remote server,
use your site's supported persistent-session or batch-job approach before starting.
Closing an SSH connection is not a reliable way to pause an analysis. Do not
launch two controllers into the same output directory at the same time.

GTDB placement may spend a long time loading a reference tree, especially in
its automatic disk-backed mode. Quiet terminal output does not by itself mean
that the process has stopped. Logs and resource reports are under `logs/<run-id>/`.
See [resources](resources.md) and [recovery](recovery.md) before stopping or restarting.

## Confirm that the run finished

Start with `Genome_inventory.tsv`, `Master_genome_QC.tsv` and
`Assessment_status.tsv`. In a successful default run, every usable input genome
should appear in the inventory and have an assessed representative in the master.
A genome with no usable sequence remains accounted for in the inventory.

The latest `logs/<run-id>/run.json` records the run's `status` and
`assessment_failures`. `complete` indicates successful completion;
`complete_with_assessment_failures` means some measurements need attention.
An explorer can be produced with incomplete assessments, so its existence alone
is not a success check. The [test checklist](test.md#check-the-results) explains
what to review before increasing the input set.

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
