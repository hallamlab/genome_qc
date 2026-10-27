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

## Choose your CPU and memory settings

| Option | Meaning | Example |
| --- | --- | --- |
| `--threads` | CPUs allocated to a parallel task; default 8 | `--threads 8` |
| `--max_cpus` | Aggregate local CPU scheduling budget | `--max_cpus 32` |
| `--max_memory` | Aggregate memory scheduling budget | `--max_memory '96 GB'` |
| `--memory` | Override each task's memory request | Usually leave unset initially |
| `--max_tasks` | Cap queued/running workflow tasks | `--max_tasks 20` |
| `--max_batch_size` | Upper bound on genomes in one balanced batch | Default 1000 |

Eight threads and 32 total CPUs allow up to four eight-CPU tasks **only when**
their memory reservations and dependencies also allow it. Setting `--max_tasks 100`
does not force 100 tasks to run on a small server. Giving a task more threads
also does not make a serial tool parallel. For GTDB placement, the disk-backed
mode has its own limit described below.

On Linux, these commands provide a quick view of your host:

```bash
nproc
free -h
df -h "$HOME"
```

Check the filesystem actually used for your databases, output and temporary work
if those paths are on different disks. Free storage needs to cover downloads,
extracted databases and analysis work simultaneously. Compact mode reduces the
completed result; it does not eliminate peak working-space requirements.
The controller's automatic memory budget does not reserve space against other
users' programs. On a shared server, set limits agreed with its administrator.

## Submit work through Slurm

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

Run the command from a host where `sbatch` is available and your cluster account
is configured. No username or password is passed to Genome QC: Slurm uses your
logged-in identity. `--account` is your allocation/project account; a partition
is a scheduler-defined collection of nodes or queue with its own limits.
Your cluster documentation tells you which account and partition to use.

The example requires writable node-local `SLURM_TMPDIR` for compact execution.
If the site does not provide it, configure an approved `--scratch_dir`; do not
assume a login-node temporary directory is visible on a worker. The installed
controller environment must also be accessible on workers. Check that database
setup and environment creation are allowed on the host running the controller;
these preparation steps need internet for anything not already installed.

The cluster's default partition is used unless `--partition` is supplied.
`--qos` and `--reservation` are also supported. Every job requests one node.
Slurm defaults to four queued-plus-running jobs and six submissions per minute;
the example deliberately raises these limits. The login node's CPU and memory
capacity does not constrain cluster jobs. Optional aggregate CPU/memory limits
conservatively reduce the job cap using the largest task request.
Tool-specific defaults are CheckM 48 GiB with up to eight threads (64 GiB above
eight), GUNC 128 GiB, GTDB-Tk 160 GiB, tRNA prediction 32 GiB, and atlas 64 GiB.
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

GTDB-Tk documents roughly 140 GB for bacterial R232 placement, with memory
varying by pplacer thread count. Genome QC reserves 160 GiB by default when
available and enables disk-backed placement below that reservation. This is a
conservative scheduling choice, not an upstream hard minimum. See the
[GTDB-Tk hardware requirements](https://ecogenomics.github.io/GTDBTk/installing/index.html#hardware-requirements).

On smaller local servers, default task reservations are capped to the usable
memory budget. The automatic aggregate budget already excludes 10% headroom;
this is not deducted a second time from individual task requests. An explicit
`--memory` request larger than the budget still produces an error.

When a GTDB task receives less than 160 GiB for R232, Genome QC automatically
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

Even a few genomes require large reference indices and tree caches to be loaded.
On slow or shared storage, this I/O can dominate runtime, especially with
disk-backed placement. A small full-workflow test is therefore not necessarily
quick. Use fast local scratch when available, allow space for the placement
files, and follow the task log and process activity rather than interpreting a
quiet placement log as a stalled run.

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

## Keep the controller within cluster limits

For large Slurm runs, use a compute-node allocation for the controller when
cluster policy permits submitting jobs from compute nodes. `--max_tasks` bounds
queued and running worker jobs; it does not cap Nextflow's JVM heap or native
threads. Genome QC reports detected controller process and cgroup limits and
warns about restrictive thread limits. This check cannot predict every runtime
limit, and a missing detected limit does not establish unlimited resources.

Download the [controller submission template](examples/controller.sbatch), activate
your installed Genome QC environment, then submit it with shared absolute paths:

```bash
sbatch --account=YOUR_ACCOUNT controller.sbatch run \
  -i /shared/project/genomes \
  -o /shared/project/genome-qc-output \
  -d /shared/project/genome-qc-databases \
  --executor slurm --account YOUR_ACCOUNT \
  --threads 8 --max_tasks 100 --submit_rate 60 \
  --time_limit 24h --compact_results
```

The template requests four controller CPUs and 32 GiB, with a default 4-GiB
Nextflow heap. Existing `NXF_OPTS` takes precedence. Adjust the controller walltime,
partition and allocation to site policy. Worker requests remain independent of
this allocation. A heap cap does not cap total JVM/native memory or thread count.

Database preparation always runs on the controller host, including with
`--executor slurm`. Prepare databases in advance with `genome_qc build_db` on a
host where downloads and extraction are permitted. Analysis reuses complete
databases. Conda environments also require controller-side preparation and
network access on first use; a compute node without network access needs those
environments prepared beforehand.

Compact mode uses worker scratch and trims disposable GTDB/GUNC products before
copying them back. Shared Nextflow checkpoints and environment caches remain
until successful completion; unresolved failures retain them for recovery.
Budget for peak usage, not only final results. Startup filesystem space/inode
counts do **not** measure user or project quotas: consult your site's quota tool.
Do not delete work directories while jobs are active. Input manifests must remain
unchanged on resume; completed compact outputs are verified against their saved
inputs, settings, implementation and product checksums.
