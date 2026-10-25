# Local and Slurm resources

All Nextflow-backed commands accept the same execution flags. Local execution
is the default: CPU capacity and physical RAM are detected, including process
affinity and cgroup limits. The automatic memory budget is 90% of that RAM
capacity, leaving headroom for the controller and operating system. It is not
based on free RAM at startup: temporary use by another process does not
permanently shrink the workflow's budget. `--threads` controls parallel tools; serial processes
request one CPU. `--max_cpus` and `--max_memory` limit aggregate scheduling,
while `--memory` overrides the memory request for each job. Memory requests
are scheduler reservations, not guaranteed application memory limits locally.
Nextflow starts queued jobs as running jobs release their CPU/memory reservations.
It does not continuously monitor unrelated applications or resize running tools.
On a shared server, use `--max_memory` and `--max_cpus` to reserve capacity for
other users; on Slurm, the cluster scheduler manages node availability.

For Slurm, use shared paths accessible from compute nodes:

```bash
genome_qc run \
  -i /shared/project/genomes \
  -o /shared/project/genome-qc-output \
  -d /shared/project/genome-qc-databases \
  --executor slurm --account YOUR_ACCOUNT \
  --threads 8 --max_tasks 100 --submit_rate 60 \
  --time_limit 24h --compact_results
```

The cluster's default partition is used unless `--partition` is supplied.
`--qos` and `--reservation` are also supported. Every job requests one node.
Slurm defaults to four queued-plus-running jobs and six submissions per minute;
the example deliberately raises these limits. The login node's CPU and memory
capacity does not constrain cluster jobs. Optional aggregate CPU/memory limits
conservatively reduce the job cap using the largest task request.
Tool-specific defaults are CheckM 48 GiB with up to eight threads (64 GiB above
eight), GUNC 128 GiB, GTDB-Tk 128 GiB, tRNA prediction 32 GiB, and atlas 64 GiB.
Small tasks request 4 GiB; merge, deduplication and quality scoring request 8 GiB.
CheckM's full reference tree requires approximately 40 GB according to its
[installation guide](https://github.com/Ecogenomics/CheckM/wiki/Installation).
The defaults add headroom but are not guarantees for every batch or thread count.
They preserve the full tree and do not change the scientific analysis.

On a 32-CPU server with 108 GiB available and `--threads 8`, two 48-GiB CheckM
batches can run concurrently if both are ready. GTDB-Tk retains its separate
larger request. CheckM, GTDB-Tk and GUNC now form deterministic balanced batches.
The planner estimates work from uncompressed FASTA file sizes, assigning larger
inputs first to the least-loaded batch. It creates enough batches for the
available per-tool concurrency, constrained by `--threads`, `--max_cpus`,
`--max_tasks` and memory reservations. This estimates work; it cannot predict
exact tool runtimes. The 1,000-genome default is a **maximum**, not a fixed block
size. Use `--max_batch_size` to lower it (never above 1,000).

For example, with 32 CPUs, 8 threads per job and about 108 GB usable memory,
two 48-GB CheckM jobs fit, so a 300-genome input is split into two batches with
similar input bytes. GTDB-Tk and GUNC each reserve most of that memory by default,
so each may have only one concurrent batch. Slurm uses the configured job limit
when no aggregate CPU/memory limit is supplied. Smaller per-tool memory overrides
can permit more jobs, but must still meet the tools' actual requirements.
GTDB-Tk stages only the genomes belonging to its batch. All GUNC batch summaries
are merged without overwriting previous batches.

Batch membership is reproducible for unchanged inputs and resource settings;
changing resource limits or the batch cap can change membership and require
those tasks to run again. Per-batch assessment receipts list the genome IDs,
and GTDB-Tk manifests are retained under `manifests/`.

Use `--checkm_memory`, `--gtdb_memory`, `--gunc_memory`, `--trnascan_memory`, or
`--atlas_memory` to override one tool, for example `--checkm_memory '64 GB'`.
These take precedence over the global `--memory` override. Explicit requests
are validated against the total budget; automatic defaults are capped as
described below. Resolved requests are printed and saved in `logs/*/run.json`.
Changes apply to the next launch, not an already-running scheduler.

Database preparation runs serially on the controller host before analysis,
and requires internet for missing references. Nextflow also prepares Mamba
environments there. Keep the controller alive for the duration of the run,
using an approved cluster session or controller job. The input, database,
output, work and environment-cache paths must be accessible from compute nodes.
Assessment tools record failures and allow independent assessments to continue.
Whole-task exits 137/143 receive one retry for the wrapped assessment processes.
Database preparation and final integration failures still stop the workflow.
See [failure handling and resume](recovery.md).

## Automatic GTDB memory handling

On smaller local servers, default task reservations are capped to the usable
memory budget. The automatic aggregate budget already excludes 10% headroom;
this is not deducted a second time from individual task requests. An explicit
`--memory` request larger than the budget still produces an error.

When a GTDB task receives less than 128 GiB for R226, Genome QC automatically
passes GTDB-Tk's `--scratch_dir` and uses up to eight pplacer CPUs, capped by
the task's CPU allocation (`--threads` and CPU budgets). Other parallel GTDB
steps continue to receive `--threads` CPUs. Larger requests use normal in-memory
placement. The choice is printed and saved in the execution receipt and task log.
The same selection applies to Slurm based on the job's memory request, without
using the login node's RAM as the cluster limit.

Placement scratch is private to the task, on its execution filesystem; it is
removed when the task exits. When Nextflow scratch is enabled, it follows that
scratch location. Merely selecting Nextflow scratch does not activate the
GTDB option independently—the memory decision does that. Disk-backed placement
can be slower and requires additional free disk space. It reduces placement
RAM use but does not impose a hard memory ceiling or guarantee that arbitrary
inputs fit. Capping other tools' default memory reservations does not reduce
their actual memory requirements.

## What runs where

Database setup and environment solving occur on the controller host. Scientific
Nextflow tasks use the selected executor. `report`, `atlas`, `compare`, `phylogeny`
and `render` are standalone commands and run in the invoking environment; they
do not inherit Slurm submission flags. On an HPC, run those commands inside an
appropriate allocation if they need substantial resources.

Inputs, reference databases, output, Nextflow work, and environment cache must be
visible to the compute nodes. Worker-local scratch is temporary execution space,
not a replacement for those shared paths. See the upstream
[Nextflow Conda guide](https://www.nextflow.io/docs/latest/conda.html) and
[Slurm executor documentation](https://www.nextflow.io/docs/latest/executor.html#slurm).
