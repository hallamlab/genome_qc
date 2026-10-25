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

After fixing a cause, use `--resume`. Failure receipts make failed assessments
eligible for retry while successful tasks remain eligible for caching. Wrapped
assessment processes receive one Nextflow retry for whole-task exit 137/143;
this does not guarantee that an out-of-memory job will fit on retry. Inspect the
logs and adjust resource requests when needed. Global setup, invalid inputs,
and final merging/reporting errors stop execution.

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
