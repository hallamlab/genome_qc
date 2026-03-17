# Genome QC Workflow (Nextflow)

## TL;DR Quick Start (New Dataset)

Use this if you want the fastest path from raw genome FASTAs to `Master_genome_QC.tsv`.

### 0) Prerequisites

- Linux shell
- `nextflow` on `PATH`
- `mamba` or `conda` on `PATH`
- Internet access for first-time environment/database setup

Quick checks:

```bash
command -v nextflow
command -v mamba || command -v conda
```

### 1) Prepare input FASTAs

Put genomes in one directory (non-recursive), for example:

```bash
/data/my_project/fasta/
```

Accepted extensions:

- `.fa`, `.fna`, `.fasta`
- `.fa.gz`, `.fna.gz`, `.fasta.gz`

### 2) Choose a reference directory

On first deployment, give the workflow a single reference directory. It will bootstrap and configure:

- CheckM reference data
- GTDB-Tk data package
- GUNC database

Example:

```bash
/data/my_project/references
```

### 3) Run pipeline on your dataset

```bash
nextflow run main.nf \
  -profile mamba \
  -name my_run_001 \
  -w /data/my_project/nf_work \
  --fasta_dir /data/my_project/fasta \
  --working_dir /data/my_project/results \
  --filter_size "2500" \
  --min_genome_size 200000 \
  --max_genome_size 15000000 \
  --completeness 0 \
  --contamination 10 \
  --qscore_min 0 \
  --checkm_cpus 2 \
  --gtdbtk_cpus 16 \
  --gunc_cpus 16 \
  --max_forks 8 \
  --ref_dir /data/my_project/references
```

### 3b) Run on Slurm HPC

Use the `slurm` and `mamba` profiles together:

```bash
nextflow run main.nf \
  -profile slurm,mamba \
  -name my_slurm_run \
  -w /scratch/$USER/genome_qc_work \
  --fasta_dir /data/my_project/fasta \
  --working_dir /data/my_project/results \
  --ref_dir /data/my_project/references \
  --slurm_account my_account \
  --slurm_partition compute \
  --slurm_time 24h \
  --checkm_cpus 4 \
  --gtdbtk_cpus 32 \
  --gunc_cpus 32
```

Optional Slurm params:

- `--slurm_account`
- `--slurm_nodes`
- `--slurm_partition`
- `--slurm_medium_partition`
- `--slurm_large_partition`
- `--slurm_qos`
- `--slurm_constraint`
- `--slurm_time`
- `--slurm_queue_size`
- `--slurm_submit_rate_limit`
- `--slurm_cluster_options`
- `--small_job_memory`, `--small_job_time`
- `--medium_job_memory`, `--medium_job_time`
- `--reference_setup_memory`, `--reference_setup_time`
- `--trnascan_memory`, `--trnascan_time`
- `--checkm_memory`, `--checkm_time`
- `--gtdbtk_memory`, `--gtdbtk_time`
- `--gunc_memory`, `--gunc_time`

### 4) Resume after interruption/failure

Use the same command with `-resume`:

```bash
nextflow run main.nf ... -resume
```

You must keep:

- same `-w` work directory
- same pipeline location
- same relevant parameters for cached steps

### 5) Key outputs

Inside `--working_dir`:

- `Master_genome_QC.tsv`
- `dedupe/fasta/*.fasta` (final deduplicated FASTA set)
- `checkm/`, `gtdbtk/`, `gunc/`, `barrnap/`, `trnascan/`, `qscore/`
- `references/` (bootstrap status tables when `--ref_dir` is used)

---

## What This Workflow Takes as Input

`main.nf` takes a FASTA directory only (`--fasta_dir`).

- It does **not** take a manifest TSV.
- It does **not** take a metadata table.

If you need manifest-driven multi-sample/category orchestration, use the wrapper scripts outside this Nextflow workflow.

---

## Parameters

All parameters are passed as `--param value`.

### Required/essential

- `--fasta_dir`: input genome FASTA directory (non-recursive)
- `--working_dir`: output directory for results
- `--ref_dir`: reference root directory for automatic first-run setup
- `--gtdbtk_data_path`: optional explicit GTDB-Tk database directory override
- `--gunc_db`: optional explicit GUNC database `.dmnd` path override
- `--checkm_data_path`: optional explicit CheckM data directory override

### Common QC settings
- `--filter_size`: minimum contig length (e.g. `"2500"`; empty means no contig cutoff)
- `--min_genome_size`
- `--max_genome_size`
- `--completeness`
- `--contamination`
- `--qscore_min`

### Performance settings

- `--checkm_cpus`: CPUs per `CHECKM_SINGLE` task
- `--checkm_batch_size`: genomes per `CHECKM_SINGLE` batch
- `--checkm_max_forks`: concurrent `CHECKM_SINGLE` batch jobs
- `--gtdbtk_cpus`: CPUs for GTDB-Tk task
- `--gtdbtk_batch_size`: genomes per GTDB-Tk batch
- `--gtdbtk_max_forks`: concurrent GTDB-Tk batch jobs
- `--gunc_cpus`: CPUs for GUNC task
- `--max_forks`: concurrency cap for per-genome processes

### Reference/bootstrap settings

- `--ref_dir`: if set, the workflow bootstraps CheckM, GTDB-Tk, and GUNC data under this directory before analysis starts
- `--checkm_data_path`: explicit CheckM data path; defaults to `<ref_dir>/checkm_data` when `--ref_dir` is used
- `--gtdbtk_data_path`: explicit GTDB-Tk data path; defaults to `<ref_dir>/gtdbtk_r220` when `--ref_dir` is used
- `--gunc_db`: explicit GUNC database file; defaults to `<ref_dir>/gunc_db/gunc_db_progenomes2.1.dmnd` when `--ref_dir` is used
- `--gtdbtk_package_url`: GTDB-Tk package URL used during bootstrap

If you point to prebuilt references instead of `--ref_dir`, the workflow now runs validation-only checks before the analysis starts and fails early if the supplied CheckM, GTDB-Tk, or GUNC paths are missing or incomplete.

For CheckM, the workflow uses `CHECKM_DATA_PATH` during task execution rather than rewriting CheckM's packaged `DATA_CONFIG`. This is more robust on HPC systems where conda envs are not writable on compute nodes.

### Slurm settings

- `--slurm_account`: Slurm account to charge
- `--slurm_nodes`: node count per submitted job; default `1`
- `--slurm_partition`: Slurm partition/queue
- `--slurm_medium_partition`: optional separate partition for moderate jobs
- `--slurm_large_partition`: optional separate partition for heavy jobs
- `--slurm_qos`: optional QoS
- `--slurm_constraint`: optional node constraint
- `--slurm_time`: walltime passed to Slurm, e.g. `24h` or `2d`
- `--slurm_queue_size`: max queued/running jobs managed by Nextflow
- `--slurm_submit_rate_limit`: submission throttle, default `10 sec`
- `--slurm_cluster_options`: raw extra Slurm options appended to each submission
- `--small_job_memory`, `--small_job_time`: default resources for low-load tasks
- `--medium_job_memory`, `--medium_job_time`: defaults for `MERGE_MASTER`, `DEDUPE`, and `QSCORE`
- `--reference_setup_memory`, `--reference_setup_time`: defaults for first-run DB bootstrap
- `--trnascan_memory`, `--trnascan_time`: defaults for `TRNASCAN`
- `--checkm_memory`, `--checkm_time`: defaults for `CHECKM_SINGLE`
- `--gtdbtk_memory`, `--gtdbtk_time`: defaults for `GTDBTK`
- `--gunc_memory`, `--gunc_time`: defaults for `GUNC`

### Debug/reporting

- `--debug true`
- profile `debug` enables Nextflow trace/timeline/report

Example:

```bash
nextflow run main.nf -profile mamba,debug ... --debug true
```

---

## Effective Runtime Behavior

- Per-genome parallel tasks:
  - `FILTER_FASTA`, `SEQKIT_STAT`, `PRUNE_FASTA`, `CHECKM_SINGLE`
  - `BARRNAP`, `BLAST_RRNA`, `TRNASCAN`
- Single aggregation tasks:
  - combine stats/checkm/GTDB-Tk, qscore, dedupe, merge master
- Batched directory-level tasks:
  - `CHECKM_SINGLE`, `GTDBTK`
- Directory-level tasks:
  - `GUNC`

CPU settings are active and respected for CheckM/GTDB-Tk/GUNC via `--checkm_cpus`, `--gtdbtk_cpus`, `--gunc_cpus`. CheckM and GTDB-Tk batching can be tuned independently with `--checkm_batch_size`, `--checkm_max_forks`, `--gtdbtk_batch_size`, and `--gtdbtk_max_forks`.

---

## Output Layout

`--working_dir` contains:

- `genomes_filtered/`
- `genomes_pruned/`
- `genomes_subset/`
- `seqkit/`
- `checkm/`
- `qscore/`
- `gtdbtk/`
- `barrnap/`
- `dedupe/`
- `dedupe/fasta/`
- `trnascan/`
- `gunc/`
- `references/` (bootstrap status tables when `--ref_dir` is used)
- `Master_genome_QC.tsv`
- summary figures from `merge_master.py`

Nextflow intermediate work files go to:

- `-w <path>` if supplied
- otherwise Nextflow default work dir

Conda env prefixes are cached at:

- `${projectDir}/.nextflow-conda` (from `nextflow.config`)

---

## Running Multiple Instances

To run multiple jobs safely at the same time:

- use a unique `-name` for each run
- use a unique `-w` per run
- use distinct `--working_dir` per run

Example:

```bash
nextflow run main.nf -profile mamba -name run_a -w /scratch/nf_run_a --working_dir /scratch/results_a ...
nextflow run main.nf -profile mamba -name run_b -w /scratch/nf_run_b --working_dir /scratch/results_b ...
```

---

## Troubleshooting

- If you see `Run name ... has been already used`, change `-name`.
- If resume is not picking up, confirm you used the same `-w` and invoke with `-resume`.
- If Conda/Mamba env creation fails (`status 143`), check for OOM or interrupted solves; re-run with `-resume` after fixing environment/cache issues.
- If GTDB-Tk fails, verify `--gtdbtk_data_path` points to an unpacked GTDB-Tk data directory, or rerun with `--ref_dir`.
- If GUNC fails, verify `--gunc_db` points to `gunc_db_progenomes2.1.dmnd`, or rerun with `--ref_dir`.
- If CheckM fails on a fresh machine, rerun with `--ref_dir` so its reference data can be bootstrapped automatically.
- If no genomes pass filters, downstream summaries may be empty.

---

## Minimal Test Run

```bash
nextflow run main.nf \
  -profile mamba \
  --fasta_dir ./test/fasta \
  --working_dir ./results_nextflow_test \
  --ref_dir ./references
```

---

## Notes

- This Nextflow workflow is maintained alongside the Snakemake workflow.
- Environments are defined in `envs/*.yaml` and created automatically by Nextflow.
