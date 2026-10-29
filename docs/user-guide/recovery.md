# Resume, failures and compact results

## Resume an interrupted run

Repeat the original command with `--resume`, keeping its input, output, database,
work and cache paths:

```bash
genome_qc run -i ~/genomes -o ~/genome-qc-output \
  -d ~/genome-qc-databases --threads 8 --max_cpus 32 --resume
```

Nextflow needs both `.genome_qc/analysis-session/.nextflow` and its task work
outputs to reuse completed work. The default work path is `.genome_qc/work`.
Do not delete them while a run is incomplete. Matching inputs, commands, helper
code and tool environments enable caching; look for `Cached process` or `CACHED`
in the trace. Changing thread counts, batch membership or tool commands may
invalidate affected tasks. Changed scientific settings require a new output
under the controller's settings guard.

## Interrupted environment creation

If a run stops while Mamba is creating a task environment, a partial environment
can remain in the cache. Nextflow may reuse that directory on resume. A missing
executable despite its declaration in the environment YAML can indicate this
case; inspect the failed task's `.command.run` for the exact environment path
and its `conda-meta/history` for the installation record.

Once no workflow is using that environment, move only the confirmed incomplete
environment out of the cache and resume. Nextflow will recreate it. Preserve
successful task work and the Nextflow session cache; do not clear the entire
output directory. Automatic repair of incomplete environment caches is not yet
implemented.

## Failed measurements are not quality exclusions

CheckM, GTDB-Tk, rRNA prediction/comparison, tRNA prediction and GUNC record
assessment failures and permit independent downstream assessments to continue.
Sequence-statistics failure does not block the other tools. A failed batch does
not contribute unverified partial metrics. Missing measurements remain missing,
not zero. A failed rRNA prediction cannot become a successful no-hit comparison.

`Assessment_status.tsv` records tool status, reason, exit code and diagnostic
log. The master table includes status columns and `assessment_complete`. The
explorer's **Incomplete assessments** count links to affected genomes. A tool
that runs but supplies no measurement is distinguished as `not_reported`.
When taxonomy is unavailable, barrnap uses its bacterial model and tRNAscan-SE
uses the general model; the fallback is recorded.

After fixing a cause in a completed run, use `--rescue` as described below.
For an interrupted workflow, `--resume` uses the original Nextflow task cache. Wrapped
assessment processes receive one Nextflow retry for whole-task exit 137/143;
this does not guarantee that an out-of-memory job will fit on retry. Inspect the
logs and adjust resource requests when needed. Global setup, invalid inputs,
and final merging/reporting errors stop execution. Missing required executables
(including Python in task environments) and unexpected assessment-wrapper
failures also stop execution; they are not treated as genome-quality outcomes.
Task environments explicitly provide the Python used by their helpers.

A final report can exist with incomplete assessments. Check the completion
message and `logs/*/run.json`, not only Nextflow's successful-task count. The
controller records `complete_with_assessment_failures` when relevant failures
remain. This is distinct from complete biological evidence for every genome.

## Compact results and checkpoints

The default keeps intermediate outputs, work directories and environment caches.
Add `--compact_results` to a full `run` for smaller completed outputs:

- Tasks execute in temporary scratch, copying their staged inputs there.
  Slurm requires writable `SLURM_TMPDIR` within each job, or an explicit
  `--scratch_dir /node/local/path`. Local execution uses `TMPDIR`, falling
  back to the system temporary directory. Reference databases remain at their
  shared paths; they are not copied into each task.
- Disposable GTDB-Tk and GUNC intermediates are removed before stage-out.
  Selected task outputs return to resumable Nextflow work storage.
- After the complete cohort, atlas and explorer succeed, redundant genome
  copies, staged inputs, Nextflow work/session caches and the private Mamba
  cache are removed. Final selected genomes, scientific tables, figures,
  input provenance and reports remain. Task command logs are archived under
  `logs/compact-task-logs.tar.gz`; execution reports and package versions remain.
- Cleanup is cohort-wide because quality batches and atlas calculations share
  inputs. Failed or interrupted workflows retain their work for `--resume`.
  Compact mode reduces temporary files but does not promise zero shared-storage
  work while the workflow is running.
- Completed compact results have a checksummed completion record. Repeating the
  command verifies and reuses them without rerunning tasks. Changed inputs,
  scientific settings or implementation require a new output directory.

`--work_dir` chooses a parent for a private per-output work directory.
`--conda_cache` chooses an explicitly shared environment cache, which is never
automatically removed. Keep those paths unchanged when resuming.
`--keep_work` makes retention explicit and cannot be combined with compact mode.
These flags do not delete the original input genomes or reference databases.

Local orchestration tests cover normal and scratch execution, cached resume,
and compact-output integrity. Actual Slurm execution still needs a cluster test.

## Common first-run problems

| What you see | What to check next |
| --- | --- |
| `genome_qc: command not found` | Activate the environment in this terminal and run `genome_qc check`. |
| No FASTA inputs found | Check the directory, supported filename extensions and file permissions; reads are not genome inputs. |
| Duplicate genome ID | Check filename stems in all nested directories, or assign unique IDs in a manifest. |
| Inputs have changed | Use a new output directory for the changed input set. |
| A task requests more memory than the budget | Check explicit global and per-tool requests against `--max_memory`; reducing a reservation does not reduce the tool's actual needs. |
| Tasks wait while one large job runs | Check CPU and memory reservations and dependencies before concluding that scheduling is stalled. |
| `python: command not found` inside a task | Check the task environment, not only Python in your terminal; see interrupted environment creation above. |
| The explorer has incomplete assessments | Inspect `Assessment_status.tsv` and the relevant task log; a missing value is not a zero measurement. |
| The browser cannot reach a remote report | Keep the report server and SSH tunnel running, and use matching port numbers. |

For an issue report, include `genome_qc --version`, the exact command, the failing
process name, its error output, and the relevant `logs/<run-id>/run.json` and
`nextflow.log`. Explain whether this was a fresh run or a resume. Keep work and
cache directories intact until the failure is understood. Task paths in the
Nextflow log point to `.command.err`, `.command.out` and `.command.sh` while
working files are retained; completed compact runs archive task logs in
`logs/compact-task-logs.tar.gz`.

## Start over after allocation-local work disappears

`--fresh_work` starts a new execution session and permits changed `--work_dir` and
`--conda_cache` locations. It is mutually exclusive with `--resume`. Inputs and
scientific settings must still match the existing output; use a new output
directory if they have changed. Existing result files are not evidence that this
new session has finished: inspect the new run receipt and final status.

For example, resubmit the same allocation script with `--fresh_work` added when
its previous `$SLURM_TMPDIR` has been removed. Subsequent `--resume` calls with
that session's work/cache locations reuse the recorded fresh session. Earlier
work directories are preserved rather than silently deleted. No option can
recover checkpoints that the cluster has already removed.

## Inspect failures and explicitly rescue a completed run

`Assessment_failures.tsv` lists failed or blocked genome assessments, diagnostic
log paths, and failure classifications. When Slurm accounting is available it
also includes job ID, scheduler state, requested CPUs/memory/time and maximum
RSS. A scheduler OOM is reported even when the biological wrapper only reports
exit code 2. Missing accounting or an ambiguous log leaves the cause `unknown`;
exit code 2 alone is not interpreted as a memory failure. Raw accounting and
failure tables are retained in each launch's `logs/<run-id>/` directory.

Repeat the original full-run command with `--rescue` and your chosen resource
changes. For example, from a project directory containing `inputs/`:

```bash
genome_qc run -i "$PWD/inputs" -o "$PWD/output" -d "$PWD/databases" \
  --executor slurm --account YOUR_ACCOUNT --threads 8 \
  --max_tasks 20 --submit_rate 6 --time_limit 24h \
  --compact_results --rescue
```

General tasks now reserve 32 GB by default; heavier tools retain higher requests.
`--memory` overrides **all** task memory requests unless a tool override is
supplied, so do not set it to 32 GB expecting GTDB-Tk to retain its larger default.
There is no automatic resource escalation or automatic rescue loop. Each rescue
is a user decision; local execution is supported too.

Rescue validates unchanged input checksums and scientific settings, then builds
an explicit reuse plan independent of Nextflow's resource-sensitive task cache.
Threads, batch sizes and resource requests may change. Keep the reference data
and existing Conda environments unchanged. Do not upgrade tools within a cached
environment: a rescue is recovery of the same analysis, not a method comparison.

RNA and preprocessing are recovered per genome. Successful rRNA prediction is
reused if only its comparison failed. Successfully completed CheckM, GTDB-Tk and
GUNC **batches** are reused; a failed native tool batch is rerun for all its
members because its partial outputs are not certified as completed assessments.
Successful taxonomy is required to reuse downstream RNA assessments; otherwise
those assessments are rerun after taxonomy recovery to use the correct model.
Final cohort tables, ANI/phylogeny products, atlas and explorer are rebuilt.

Each rescue retains its plan, previous assessment table and failed diagnostics
under `.genome_qc/rescue/<run-id>/`. Results must have the expected products to be
reused; missing products are scheduled again. Legacy 0.1.1 unpublished RNA and
preprocessing outputs can be recovered from the retained work directories.
Do not delete work or cache directories before rescue. Compact cleanup remains
deferred while failures remain. `--rescue` requires a previous final assessment
table and cannot be combined with `--resume` or `--fresh_work`.
