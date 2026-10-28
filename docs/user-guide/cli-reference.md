# CLI reference

Generated from the command parsers by `scripts/build_cli_reference.py`.
Every public subcommand is included, including the nested tree-render commands.
See [commands](commands.md) for choosing a module, [resources](resources.md) for
runtime resource defaults, [selection](selection.md) for acceptance criteria,
and the [output dictionary](output-columns.md) for the resulting measurements.

## Common option meanings

Options apply only where shown in the corresponding help block. The complete
workflow uses underscore-style flags; standalone atlas/comparison/render commands
use their own hyphen-style flags. These are not interchangeable aliases.

| Option | Meaning / default |
| --- | --- |
| `-i`, `--input` | Genome FASTA directory, searched recursively. Mutually exclusive with `--manifest`. |
| `--manifest` | TSV containing unique `id` and `fasta`; relative FASTA paths resolve beside the manifest. Extra fields are retained as metadata. |
| `-o`, `--output` | Required analysis output directory. Keep outside the recursively searched input tree. Standalone analyses may call this `--output-dir`; read their help. |
| `-d`, `--database` | Reference root; default `~/.local/share/genome_qc/databases`. Complete existing references are reused. |
| `--threads` | CPUs requested per parallel tool (default 8 for workflow stages); not an aggregate machine/job limit. See `--max_cpus`. |
| `--resume` | Ask Nextflow to reuse matching cached work. Requires the original inputs/settings and work/cache locations. Completed compact results are verified/reused without needing task caches. |
| `--plan` | Validate/stage inputs and save commands/settings without launching Nextflow or downloading references. Creates files: use a disposable planning output directory. |
| `--params` | Advanced Nextflow parameters in a JSON object. Explicit CLI selection/resource settings can override corresponding values; inspect the saved `params.json`. This is not a free-form shell config. |
| `--filter_size` | Contig minimum-length variants, e.g. `--filter_size "1000 2500"`; absent means all contigs (`full`). This creates variants; final acceptance thresholds are separate. |
| `--kingdom` | RNA-model choice for standalone RNA commands; default Bacteria. The complete workflow uses available GTDB taxonomy with recorded fallbacks. |
| `--serve` | Start the local report HTTP server; keep the terminal alive while browsing. |
| `--port` | Report port, default 8765. For remote use, match the SSH forwarding port. |
| `--version` | Print controller version and exit. |
| `-h`, `--help` | Print help for the current command and exit. |

`check` checks availability of Nextflow, Java and Mamba on PATH; it does not
download databases or validate the scientific tools. `--max_tasks` caps queued
and running worker tasks, not controller threads or heap size.

**Different ANI defaults:** `run` defaults to 98% ANI and larger-assembly AF 0.51
with labelled containment rescue. Standalone `atlas` / `compare` retain their
comparison defaults (95% ANI, AF 0.5) and their separate comparison logic. Do not
assume they reproduce `run` cohort groups without reviewing their configuration.

## Top-level command

```text
usage: genome_qc [-h] [--version]
                 {run,build_db,preprocess,quality,taxonomy,rrna,trna,chimera,atlas,compare,phylogeny,render,check,report}
                 ...

Genome quality assessment and genome atlas

positional arguments:
  {run,build_db,preprocess,quality,taxonomy,rrna,trna,chimera,atlas,compare,phylogeny,render,check,report}
    atlas               atlas analysis; use atlas --help for options
    compare             compare analysis; use compare --help for options
    phylogeny           phylogeny analysis; use phylogeny --help for options
    render              render analysis; use render --help for options
    check               Check controller dependencies without downloading data
    report              Index output tables and open the explorer

options:
  -h, --help            show this help message and exit
  --version             show program's version number and exit
```

## run

```text
usage: genome_qc run [-h] [-i INPUT] [--no_recursive] [--manifest MANIFEST] -o
                     OUTPUT [-d DATABASE] [--threads THREADS]
                     [--max_cpus MAX_CPUS] [--max_memory MAX_MEMORY]
                     [--memory MEMORY] [--checkm_memory CHECKM_MEMORY]
                     [--gtdb_memory GTDB_MEMORY] [--gunc_memory GUNC_MEMORY]
                     [--trnascan_memory TRNASCAN_MEMORY]
                     [--atlas_memory ATLAS_MEMORY] [--max_tasks MAX_TASKS]
                     [--executor {local,slurm}] [--account ACCOUNT]
                     [--partition PARTITION] [--qos QOS]
                     [--reservation RESERVATION] [--time_limit TIME_LIMIT]
                     [--submit_rate SUBMIT_RATE] [--work_dir WORK_DIR]
                     [--conda_cache CONDA_CACHE] [--keep_work]
                     [--compact_results] [--scratch_dir SCRATCH_DIR]
                     [--resume | --fresh_work]
                     [--small_batch_size SMALL_BATCH_SIZE]
                     [--file_budget FILE_BUDGET] [--plan] [--params PARAMS]
                     [--max_batch_size MAX_BATCH_SIZE]
                     [--filter_size FILTER_SIZE]
                     [--min_genome_size MIN_GENOME_SIZE]
                     [--max_genome_size MAX_GENOME_SIZE]
                     [--completeness COMPLETENESS]
                     [--contamination CONTAMINATION] [--qscore_min QSCORE_MIN]
                     [--require_rrna_consistency]
                     [--ani_threshold ANI_THRESHOLD]
                     [--ani_af_threshold ANI_AF_THRESHOLD]
                     [--no_ani_containment] [--skip_ani] [--skip_phylogeny]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Genome FASTA directory (searched recursively)
  --no_recursive        Only discover FASTAs directly inside -i; do not search
                        subdirectories
  --manifest MANIFEST   TSV with id and fasta; relative paths resolve beside
                        the manifest
  -o OUTPUT, --output OUTPUT
  -d DATABASE, --database DATABASE
  --threads THREADS
  --resume
  --fresh_work          Start a new execution session with unchanged
                        inputs/settings; permit changed work/cache paths
  --small_batch_size SMALL_BATCH_SIZE
                        Maximum genomes per lightweight task batch [128]
  --file_budget FILE_BUDGET
                        Available file quota for this run; warn if the
                        planning envelope exceeds it
  --plan                Validate inputs and save commands without executing
  --params PARAMS       Advanced Nextflow parameters as a JSON object
  --max_batch_size MAX_BATCH_SIZE
                        Maximum genomes per balanced batch (default: 1000);
                        smaller datasets use resource-aware batches

Execution and resources:
  --max_cpus MAX_CPUS   Total CPU budget; local defaults to available CPUs,
                        Slurm has no default aggregate cap
  --max_memory MAX_MEMORY
                        Total memory budget; local defaults to 90% of
                        physical/cgroup/Slurm capacity, not free RAM at
                        startup
  --memory MEMORY       Override memory requested per task unless a tool-
                        specific override is supplied
  --checkm_memory CHECKM_MEMORY
                        Override checkm task memory independently of other
                        tools
  --gtdb_memory GTDB_MEMORY
                        Override gtdb task memory independently of other tools
  --gunc_memory GUNC_MEMORY
                        Override gunc task memory independently of other tools
  --trnascan_memory TRNASCAN_MEMORY
                        Override trnascan task memory independently of other
                        tools
  --atlas_memory ATLAS_MEMORY
                        Override atlas task memory independently of other
                        tools
  --max_tasks MAX_TASKS
                        Queued plus running tasks [local: CPU budget; Slurm:
                        4]
  --executor {local,slurm}
  --account ACCOUNT
  --partition PARTITION
  --qos QOS
  --reservation RESERVATION
  --time_limit TIME_LIMIT
  --submit_rate SUBMIT_RATE
                        Slurm submissions per minute [6]
  --work_dir WORK_DIR   Custom work parent; only this run’s private child is
                        managed
  --conda_cache CONDA_CACHE
                        Shared environment cache; never deleted automatically
  --keep_work           Retain work; incompatible with compact output
  --compact_results     Use task scratch and compact successful full-workflow
                        outputs
  --scratch_dir SCRATCH_DIR
                        Worker-local scratch base; otherwise SLURM_TMPDIR on
                        Slurm, TMPDIR locally

Preprocessing:
  --filter_size FILTER_SIZE
                        Contig-length variants; default: retain all contigs

Final selection (default: retain every assessed genome):
  --min_genome_size MIN_GENOME_SIZE
                        Minimum genome length (bp); default: unrestricted
  --max_genome_size MAX_GENOME_SIZE
                        Maximum genome length (bp); default: unrestricted
  --completeness COMPLETENESS
                        Minimum CheckM completeness (%); default: unrestricted
  --contamination CONTAMINATION
                        Maximum CheckM contamination (%); default:
                        unrestricted
  --qscore_min QSCORE_MIN
                        Minimum Qscore; default: unrestricted
  --require_rrna_consistency
                        Require the rRNA consistency check to pass; default:
                        disabled

Atlas:
  --ani_threshold ANI_THRESHOLD
                        ANI grouping threshold (%); default: 98
  --ani_af_threshold ANI_AF_THRESHOLD
                        ANI alignment fraction (0–1); default: 0.51
  --no_ani_containment  Disable containment rescue (default: smaller AF >=0.90
                        and larger AF >=0.10)
  --skip_ani            Skip cohort ANI grouping
  --skip_phylogeny      Skip GTDB-marker phylogeny and annotated trees
```

## build_db

```text
usage: genome_qc build_db [-h] -o OUTPUT [-d DATABASE] [--threads THREADS]
                          [--max_cpus MAX_CPUS] [--max_memory MAX_MEMORY]
                          [--memory MEMORY] [--max_tasks MAX_TASKS]
                          [--executor {local,slurm}] [--account ACCOUNT]
                          [--partition PARTITION] [--qos QOS]
                          [--reservation RESERVATION]
                          [--time_limit TIME_LIMIT]
                          [--submit_rate SUBMIT_RATE] [--work_dir WORK_DIR]
                          [--conda_cache CONDA_CACHE] [--keep_work]
                          [--compact_results] [--scratch_dir SCRATCH_DIR]
                          [--resume | --fresh_work]
                          [--download_only | --extract_only]
                          [--download_attempts DOWNLOAD_ATTEMPTS]
                          [--db_url NAME=URL] [--plan] [--params PARAMS]

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
  -d DATABASE, --database DATABASE
  --threads THREADS
  --resume
  --fresh_work          Start a new execution session with unchanged
                        inputs/settings; permit changed work/cache paths
  --download_only       Fetch verified reference archives without installing
  --extract_only        Install supplied archives without downloading
  --download_attempts DOWNLOAD_ATTEMPTS
  --db_url NAME=URL     Alternate archive URL for checkm, gtdbtk or gunc;
                        pinned checksum remains required
  --plan                Validate inputs and save commands without executing
  --params PARAMS       Advanced Nextflow parameters as a JSON object

Execution and resources:
  --max_cpus MAX_CPUS   Total CPU budget; local defaults to available CPUs,
                        Slurm has no default aggregate cap
  --max_memory MAX_MEMORY
                        Total memory budget; local defaults to 90% of
                        physical/cgroup/Slurm capacity, not free RAM at
                        startup
  --memory MEMORY       Override memory requested per task unless a tool-
                        specific override is supplied
  --max_tasks MAX_TASKS
                        Queued plus running tasks [local: CPU budget; Slurm:
                        4]
  --executor {local,slurm}
  --account ACCOUNT
  --partition PARTITION
  --qos QOS
  --reservation RESERVATION
  --time_limit TIME_LIMIT
  --submit_rate SUBMIT_RATE
                        Slurm submissions per minute [6]
  --work_dir WORK_DIR   Custom work parent; only this run’s private child is
                        managed
  --conda_cache CONDA_CACHE
                        Shared environment cache; never deleted automatically
  --keep_work           Retain work; incompatible with compact output
  --compact_results     Use task scratch and compact successful full-workflow
                        outputs
  --scratch_dir SCRATCH_DIR
                        Worker-local scratch base; otherwise SLURM_TMPDIR on
                        Slurm, TMPDIR locally
```

## preprocess

```text
usage: genome_qc preprocess [-h] [-i INPUT] [--no_recursive]
                            [--manifest MANIFEST] -o OUTPUT [-d DATABASE]
                            [--threads THREADS] [--max_cpus MAX_CPUS]
                            [--max_memory MAX_MEMORY] [--memory MEMORY]
                            [--max_tasks MAX_TASKS] [--executor {local,slurm}]
                            [--account ACCOUNT] [--partition PARTITION]
                            [--qos QOS] [--reservation RESERVATION]
                            [--time_limit TIME_LIMIT]
                            [--submit_rate SUBMIT_RATE] [--work_dir WORK_DIR]
                            [--conda_cache CONDA_CACHE] [--keep_work]
                            [--compact_results] [--scratch_dir SCRATCH_DIR]
                            [--resume | --fresh_work] [--plan]
                            [--params PARAMS] [--filter_size FILTER_SIZE]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Genome FASTA directory (searched recursively)
  --no_recursive        Only discover FASTAs directly inside -i; do not search
                        subdirectories
  --manifest MANIFEST   TSV with id and fasta; relative paths resolve beside
                        the manifest
  -o OUTPUT, --output OUTPUT
  -d DATABASE, --database DATABASE
  --threads THREADS
  --resume
  --fresh_work          Start a new execution session with unchanged
                        inputs/settings; permit changed work/cache paths
  --plan                Validate inputs and save commands without executing
  --params PARAMS       Advanced Nextflow parameters as a JSON object

Execution and resources:
  --max_cpus MAX_CPUS   Total CPU budget; local defaults to available CPUs,
                        Slurm has no default aggregate cap
  --max_memory MAX_MEMORY
                        Total memory budget; local defaults to 90% of
                        physical/cgroup/Slurm capacity, not free RAM at
                        startup
  --memory MEMORY       Override memory requested per task unless a tool-
                        specific override is supplied
  --max_tasks MAX_TASKS
                        Queued plus running tasks [local: CPU budget; Slurm:
                        4]
  --executor {local,slurm}
  --account ACCOUNT
  --partition PARTITION
  --qos QOS
  --reservation RESERVATION
  --time_limit TIME_LIMIT
  --submit_rate SUBMIT_RATE
                        Slurm submissions per minute [6]
  --work_dir WORK_DIR   Custom work parent; only this run’s private child is
                        managed
  --conda_cache CONDA_CACHE
                        Shared environment cache; never deleted automatically
  --keep_work           Retain work; incompatible with compact output
  --compact_results     Use task scratch and compact successful full-workflow
                        outputs
  --scratch_dir SCRATCH_DIR
                        Worker-local scratch base; otherwise SLURM_TMPDIR on
                        Slurm, TMPDIR locally

Preprocessing:
  --filter_size FILTER_SIZE
                        Contig-length variants; default: retain all contigs
```

## quality

```text
usage: genome_qc quality [-h] [-i INPUT] [--no_recursive]
                         [--manifest MANIFEST] -o OUTPUT [-d DATABASE]
                         [--threads THREADS] [--max_cpus MAX_CPUS]
                         [--max_memory MAX_MEMORY] [--memory MEMORY]
                         [--checkm_memory CHECKM_MEMORY]
                         [--max_tasks MAX_TASKS] [--executor {local,slurm}]
                         [--account ACCOUNT] [--partition PARTITION]
                         [--qos QOS] [--reservation RESERVATION]
                         [--time_limit TIME_LIMIT] [--submit_rate SUBMIT_RATE]
                         [--work_dir WORK_DIR] [--conda_cache CONDA_CACHE]
                         [--keep_work] [--compact_results]
                         [--scratch_dir SCRATCH_DIR] [--resume | --fresh_work]
                         [--plan] [--params PARAMS]
                         [--max_batch_size MAX_BATCH_SIZE]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Genome FASTA directory (searched recursively)
  --no_recursive        Only discover FASTAs directly inside -i; do not search
                        subdirectories
  --manifest MANIFEST   TSV with id and fasta; relative paths resolve beside
                        the manifest
  -o OUTPUT, --output OUTPUT
  -d DATABASE, --database DATABASE
  --threads THREADS
  --resume
  --fresh_work          Start a new execution session with unchanged
                        inputs/settings; permit changed work/cache paths
  --plan                Validate inputs and save commands without executing
  --params PARAMS       Advanced Nextflow parameters as a JSON object
  --max_batch_size MAX_BATCH_SIZE
                        Maximum genomes per balanced batch (default: 1000);
                        smaller datasets use resource-aware batches

Execution and resources:
  --max_cpus MAX_CPUS   Total CPU budget; local defaults to available CPUs,
                        Slurm has no default aggregate cap
  --max_memory MAX_MEMORY
                        Total memory budget; local defaults to 90% of
                        physical/cgroup/Slurm capacity, not free RAM at
                        startup
  --memory MEMORY       Override memory requested per task unless a tool-
                        specific override is supplied
  --checkm_memory CHECKM_MEMORY
                        Override checkm task memory independently of other
                        tools
  --max_tasks MAX_TASKS
                        Queued plus running tasks [local: CPU budget; Slurm:
                        4]
  --executor {local,slurm}
  --account ACCOUNT
  --partition PARTITION
  --qos QOS
  --reservation RESERVATION
  --time_limit TIME_LIMIT
  --submit_rate SUBMIT_RATE
                        Slurm submissions per minute [6]
  --work_dir WORK_DIR   Custom work parent; only this run’s private child is
                        managed
  --conda_cache CONDA_CACHE
                        Shared environment cache; never deleted automatically
  --keep_work           Retain work; incompatible with compact output
  --compact_results     Use task scratch and compact successful full-workflow
                        outputs
  --scratch_dir SCRATCH_DIR
                        Worker-local scratch base; otherwise SLURM_TMPDIR on
                        Slurm, TMPDIR locally
```

## taxonomy

```text
usage: genome_qc taxonomy [-h] [-i INPUT] [--no_recursive]
                          [--manifest MANIFEST] -o OUTPUT [-d DATABASE]
                          [--threads THREADS] [--max_cpus MAX_CPUS]
                          [--max_memory MAX_MEMORY] [--memory MEMORY]
                          [--gtdb_memory GTDB_MEMORY] [--max_tasks MAX_TASKS]
                          [--executor {local,slurm}] [--account ACCOUNT]
                          [--partition PARTITION] [--qos QOS]
                          [--reservation RESERVATION]
                          [--time_limit TIME_LIMIT]
                          [--submit_rate SUBMIT_RATE] [--work_dir WORK_DIR]
                          [--conda_cache CONDA_CACHE] [--keep_work]
                          [--compact_results] [--scratch_dir SCRATCH_DIR]
                          [--resume | --fresh_work] [--plan] [--params PARAMS]
                          [--max_batch_size MAX_BATCH_SIZE]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Genome FASTA directory (searched recursively)
  --no_recursive        Only discover FASTAs directly inside -i; do not search
                        subdirectories
  --manifest MANIFEST   TSV with id and fasta; relative paths resolve beside
                        the manifest
  -o OUTPUT, --output OUTPUT
  -d DATABASE, --database DATABASE
  --threads THREADS
  --resume
  --fresh_work          Start a new execution session with unchanged
                        inputs/settings; permit changed work/cache paths
  --plan                Validate inputs and save commands without executing
  --params PARAMS       Advanced Nextflow parameters as a JSON object
  --max_batch_size MAX_BATCH_SIZE
                        Maximum genomes per balanced batch (default: 1000);
                        smaller datasets use resource-aware batches

Execution and resources:
  --max_cpus MAX_CPUS   Total CPU budget; local defaults to available CPUs,
                        Slurm has no default aggregate cap
  --max_memory MAX_MEMORY
                        Total memory budget; local defaults to 90% of
                        physical/cgroup/Slurm capacity, not free RAM at
                        startup
  --memory MEMORY       Override memory requested per task unless a tool-
                        specific override is supplied
  --gtdb_memory GTDB_MEMORY
                        Override gtdb task memory independently of other tools
  --max_tasks MAX_TASKS
                        Queued plus running tasks [local: CPU budget; Slurm:
                        4]
  --executor {local,slurm}
  --account ACCOUNT
  --partition PARTITION
  --qos QOS
  --reservation RESERVATION
  --time_limit TIME_LIMIT
  --submit_rate SUBMIT_RATE
                        Slurm submissions per minute [6]
  --work_dir WORK_DIR   Custom work parent; only this run’s private child is
                        managed
  --conda_cache CONDA_CACHE
                        Shared environment cache; never deleted automatically
  --keep_work           Retain work; incompatible with compact output
  --compact_results     Use task scratch and compact successful full-workflow
                        outputs
  --scratch_dir SCRATCH_DIR
                        Worker-local scratch base; otherwise SLURM_TMPDIR on
                        Slurm, TMPDIR locally
```

## rrna

```text
usage: genome_qc rrna [-h] [-i INPUT] [--no_recursive] [--manifest MANIFEST]
                      -o OUTPUT [-d DATABASE] [--threads THREADS]
                      [--max_cpus MAX_CPUS] [--max_memory MAX_MEMORY]
                      [--memory MEMORY] [--max_tasks MAX_TASKS]
                      [--executor {local,slurm}] [--account ACCOUNT]
                      [--partition PARTITION] [--qos QOS]
                      [--reservation RESERVATION] [--time_limit TIME_LIMIT]
                      [--submit_rate SUBMIT_RATE] [--work_dir WORK_DIR]
                      [--conda_cache CONDA_CACHE] [--keep_work]
                      [--compact_results] [--scratch_dir SCRATCH_DIR]
                      [--resume | --fresh_work] [--plan] [--params PARAMS]
                      [--kingdom {Bacteria,Archaea,Eukaryota}]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Genome FASTA directory (searched recursively)
  --no_recursive        Only discover FASTAs directly inside -i; do not search
                        subdirectories
  --manifest MANIFEST   TSV with id and fasta; relative paths resolve beside
                        the manifest
  -o OUTPUT, --output OUTPUT
  -d DATABASE, --database DATABASE
  --threads THREADS
  --resume
  --fresh_work          Start a new execution session with unchanged
                        inputs/settings; permit changed work/cache paths
  --plan                Validate inputs and save commands without executing
  --params PARAMS       Advanced Nextflow parameters as a JSON object
  --kingdom {Bacteria,Archaea,Eukaryota}

Execution and resources:
  --max_cpus MAX_CPUS   Total CPU budget; local defaults to available CPUs,
                        Slurm has no default aggregate cap
  --max_memory MAX_MEMORY
                        Total memory budget; local defaults to 90% of
                        physical/cgroup/Slurm capacity, not free RAM at
                        startup
  --memory MEMORY       Override memory requested per task unless a tool-
                        specific override is supplied
  --max_tasks MAX_TASKS
                        Queued plus running tasks [local: CPU budget; Slurm:
                        4]
  --executor {local,slurm}
  --account ACCOUNT
  --partition PARTITION
  --qos QOS
  --reservation RESERVATION
  --time_limit TIME_LIMIT
  --submit_rate SUBMIT_RATE
                        Slurm submissions per minute [6]
  --work_dir WORK_DIR   Custom work parent; only this run’s private child is
                        managed
  --conda_cache CONDA_CACHE
                        Shared environment cache; never deleted automatically
  --keep_work           Retain work; incompatible with compact output
  --compact_results     Use task scratch and compact successful full-workflow
                        outputs
  --scratch_dir SCRATCH_DIR
                        Worker-local scratch base; otherwise SLURM_TMPDIR on
                        Slurm, TMPDIR locally
```

## trna

```text
usage: genome_qc trna [-h] [-i INPUT] [--no_recursive] [--manifest MANIFEST]
                      -o OUTPUT [-d DATABASE] [--threads THREADS]
                      [--max_cpus MAX_CPUS] [--max_memory MAX_MEMORY]
                      [--memory MEMORY] [--trnascan_memory TRNASCAN_MEMORY]
                      [--max_tasks MAX_TASKS] [--executor {local,slurm}]
                      [--account ACCOUNT] [--partition PARTITION] [--qos QOS]
                      [--reservation RESERVATION] [--time_limit TIME_LIMIT]
                      [--submit_rate SUBMIT_RATE] [--work_dir WORK_DIR]
                      [--conda_cache CONDA_CACHE] [--keep_work]
                      [--compact_results] [--scratch_dir SCRATCH_DIR]
                      [--resume | --fresh_work] [--plan] [--params PARAMS]
                      [--kingdom {Bacteria,Archaea,Eukaryota}]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Genome FASTA directory (searched recursively)
  --no_recursive        Only discover FASTAs directly inside -i; do not search
                        subdirectories
  --manifest MANIFEST   TSV with id and fasta; relative paths resolve beside
                        the manifest
  -o OUTPUT, --output OUTPUT
  -d DATABASE, --database DATABASE
  --threads THREADS
  --resume
  --fresh_work          Start a new execution session with unchanged
                        inputs/settings; permit changed work/cache paths
  --plan                Validate inputs and save commands without executing
  --params PARAMS       Advanced Nextflow parameters as a JSON object
  --kingdom {Bacteria,Archaea,Eukaryota}

Execution and resources:
  --max_cpus MAX_CPUS   Total CPU budget; local defaults to available CPUs,
                        Slurm has no default aggregate cap
  --max_memory MAX_MEMORY
                        Total memory budget; local defaults to 90% of
                        physical/cgroup/Slurm capacity, not free RAM at
                        startup
  --memory MEMORY       Override memory requested per task unless a tool-
                        specific override is supplied
  --trnascan_memory TRNASCAN_MEMORY
                        Override trnascan task memory independently of other
                        tools
  --max_tasks MAX_TASKS
                        Queued plus running tasks [local: CPU budget; Slurm:
                        4]
  --executor {local,slurm}
  --account ACCOUNT
  --partition PARTITION
  --qos QOS
  --reservation RESERVATION
  --time_limit TIME_LIMIT
  --submit_rate SUBMIT_RATE
                        Slurm submissions per minute [6]
  --work_dir WORK_DIR   Custom work parent; only this run’s private child is
                        managed
  --conda_cache CONDA_CACHE
                        Shared environment cache; never deleted automatically
  --keep_work           Retain work; incompatible with compact output
  --compact_results     Use task scratch and compact successful full-workflow
                        outputs
  --scratch_dir SCRATCH_DIR
                        Worker-local scratch base; otherwise SLURM_TMPDIR on
                        Slurm, TMPDIR locally
```

## chimera

```text
usage: genome_qc chimera [-h] [-i INPUT] [--no_recursive]
                         [--manifest MANIFEST] -o OUTPUT [-d DATABASE]
                         [--threads THREADS] [--max_cpus MAX_CPUS]
                         [--max_memory MAX_MEMORY] [--memory MEMORY]
                         [--gunc_memory GUNC_MEMORY] [--max_tasks MAX_TASKS]
                         [--executor {local,slurm}] [--account ACCOUNT]
                         [--partition PARTITION] [--qos QOS]
                         [--reservation RESERVATION] [--time_limit TIME_LIMIT]
                         [--submit_rate SUBMIT_RATE] [--work_dir WORK_DIR]
                         [--conda_cache CONDA_CACHE] [--keep_work]
                         [--compact_results] [--scratch_dir SCRATCH_DIR]
                         [--resume | --fresh_work] [--plan] [--params PARAMS]
                         [--max_batch_size MAX_BATCH_SIZE]

options:
  -h, --help            show this help message and exit
  -i INPUT, --input INPUT
                        Genome FASTA directory (searched recursively)
  --no_recursive        Only discover FASTAs directly inside -i; do not search
                        subdirectories
  --manifest MANIFEST   TSV with id and fasta; relative paths resolve beside
                        the manifest
  -o OUTPUT, --output OUTPUT
  -d DATABASE, --database DATABASE
  --threads THREADS
  --resume
  --fresh_work          Start a new execution session with unchanged
                        inputs/settings; permit changed work/cache paths
  --plan                Validate inputs and save commands without executing
  --params PARAMS       Advanced Nextflow parameters as a JSON object
  --max_batch_size MAX_BATCH_SIZE
                        Maximum genomes per balanced batch (default: 1000);
                        smaller datasets use resource-aware batches

Execution and resources:
  --max_cpus MAX_CPUS   Total CPU budget; local defaults to available CPUs,
                        Slurm has no default aggregate cap
  --max_memory MAX_MEMORY
                        Total memory budget; local defaults to 90% of
                        physical/cgroup/Slurm capacity, not free RAM at
                        startup
  --memory MEMORY       Override memory requested per task unless a tool-
                        specific override is supplied
  --gunc_memory GUNC_MEMORY
                        Override gunc task memory independently of other tools
  --max_tasks MAX_TASKS
                        Queued plus running tasks [local: CPU budget; Slurm:
                        4]
  --executor {local,slurm}
  --account ACCOUNT
  --partition PARTITION
  --qos QOS
  --reservation RESERVATION
  --time_limit TIME_LIMIT
  --submit_rate SUBMIT_RATE
                        Slurm submissions per minute [6]
  --work_dir WORK_DIR   Custom work parent; only this run’s private child is
                        managed
  --conda_cache CONDA_CACHE
                        Shared environment cache; never deleted automatically
  --keep_work           Retain work; incompatible with compact output
  --compact_results     Use task scratch and compact successful full-workflow
                        outputs
  --scratch_dir SCRATCH_DIR
                        Worker-local scratch base; otherwise SLURM_TMPDIR on
                        Slurm, TMPDIR locally
```

## report

```text
usage: genome_qc report [-h] -o OUTPUT [--serve] [--port PORT]

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
  --serve
  --port PORT
```

## check

```text
usage: genome_qc check [-h]

options:
  -h, --help  show this help message and exit
```

## atlas

```text
usage: genome_qc atlas [-h] [-o OUTPUT_DIR] [--prefix PREFIX]
                               [--group-column GROUP_COLUMN]
                               [--sample-column SAMPLE_COLUMN]
                               [--matched-samples-only]
                               [--top-n-groups TOP_N_GROUPS]
                               [--compare-column COMPARE_COLUMN]
                               [--compare-map COMPARE_MAP]
                               [--compare-map-key COMPARE_MAP_KEY]
                               [--ani-compare-column ANI_COMPARE_COLUMN]
                               [--ani-results ANI_RESULTS]
                               [--ani-genome-dir ANI_GENOME_DIR]
                               [--ani-fasta-column ANI_FASTA_COLUMN]
                               [--ani-fasta-exts ANI_FASTA_EXTS]
                               [--ani-threshold ANI_THRESHOLD]
                               [--ani-af-threshold ANI_AF_THRESHOLD]
                               [--ani-threads ANI_THREADS]
                               [--ani-top-overlaps ANI_TOP_OVERLAPS]
                               master_tsv

Compute a continuous MIMAG-style genome quality index and generate a multi-
panel atlas from Master_genome_QC.tsv.

positional arguments:
  master_tsv            Path to the master genome QC TSV file.

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory. Defaults to <master-
                        dir>/genome_quality_atlas.
  --prefix PREFIX       Prefix for output files. Default: genome_quality
  --group-column GROUP_COLUMN
                        Optional column used for group summaries/statistics
                        and the atlas comparison panel. Example: Domain,
                        Phylum, sample_set.
  --sample-column SAMPLE_COLUMN
                        Optional sample/replicate column used for sample-aware
                        category count comparisons. Example: sample, site,
                        replicate.
  --matched-samples-only
                        Restrict category-comparison outputs to samples
                        present in every category. FastANI overlap/dedup
                        comparisons are then constrained to within-sample
                        matches only.
  --top-n-groups TOP_N_GROUPS
                        Maximum number of groups to show in the comparison
                        panel. Default: 12
  --compare-column COMPARE_COLUMN
                        Category column to compare across. May be passed
                        multiple times. Columns can come from the master table
                        or from --compare-map.
  --compare-map COMPARE_MAP
                        Optional TSV/CSV containing Genome_Id or Bin Id plus
                        one or more custom category columns to merge before
                        comparison plotting.
  --compare-map-key COMPARE_MAP_KEY
                        Join key for --compare-map. Defaults to auto-detecting
                        Genome_Id or Bin Id.
  --ani-compare-column ANI_COMPARE_COLUMN
                        Category column for FastANI-based overlap comparison.
                        May be passed multiple times.
  --ani-results ANI_RESULTS
                        Optional existing FastANI results file to reuse
                        instead of running FastANI.
  --ani-genome-dir ANI_GENOME_DIR
                        Directory containing genome FASTA files. Used when
                        running FastANI unless --ani-fasta-column is provided.
  --ani-fasta-column ANI_FASTA_COLUMN
                        Column containing genome FASTA paths for FastANI.
  --ani-fasta-exts ANI_FASTA_EXTS
                        Comma-separated FASTA extensions used when resolving
                        files from --ani-genome-dir.
  --ani-threshold ANI_THRESHOLD
                        ANI threshold for calling genomes matched. Default:
                        95.0
  --ani-af-threshold ANI_AF_THRESHOLD
                        Minimum alignment fraction for calling genomes
                        matched. Default: 0.5
  --ani-threads ANI_THREADS
                        Number of threads for FastANI. Default: 1
  --ani-top-overlaps ANI_TOP_OVERLAPS
                        Maximum number of overlap combinations to show in the
                        UpSet plot. Default: 15
```

## compare

```text
usage: genome_qc compare [-h] [-o OUTPUT_DIR]
                                       [--atlas-script ATLAS_SCRIPT]
                                       [--sample-column SAMPLE_COLUMN]
                                       [--category-column CATEGORY_COLUMN]
                                       [--group-column GROUP_COLUMN]
                                       [--top-n-groups TOP_N_GROUPS]
                                       [--individual-output-name INDIVIDUAL_OUTPUT_NAME]
                                       [--individual-prefix INDIVIDUAL_PREFIX]
                                       [--global-prefix GLOBAL_PREFIX]
                                       [--individual-master-name INDIVIDUAL_MASTER_NAME]
                                       [--combined-master-name COMBINED_MASTER_NAME]
                                       [--ani-threshold ANI_THRESHOLD]
                                       [--ani-af-threshold ANI_AF_THRESHOLD]
                                       [--ani-threads ANI_THREADS]
                                       [--ani-top-overlaps ANI_TOP_OVERLAPS]
                                       [--ani-results ANI_RESULTS]
                                       [--matched-samples-only]
                                       [--best-sample-dir-name BEST_SAMPLE_DIR_NAME]
                                       [--best-global-dir-name BEST_GLOBAL_DIR_NAME]
                                       [--best-set-min-mimag-tier {low,medium,high}]
                                       [--skip-to-best]
                                       [--skip-organize-outputs]
                                       [--run-best-set-phylogeny]
                                       [--best-set-phylogeny-threads BEST_SET_PHYLOGENY_THREADS]
                                       [--best-set-gtdbtk-data-path BEST_SET_GTDBTK_DATA_PATH]
                                       manifest_tsv

Run genome_quality_atlas.py across multiple run directories, build a combined
master table, and perform a global category comparison.

positional arguments:
  manifest_tsv          Preferred manifest TSV: sample, category,
                        run_directory, bin_id_token_index. The fourth column
                        is optional and is a 1-based period-delimited token
                        index used to pre-deduplicate same-bin
                        parameterization variants within each sample/category.
                        A legacy two-column form (category, run_directory) is
                        also accepted.

options:
  -h, --help            show this help message and exit
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Output directory for the combined master and global
                        atlas. Defaults to <manifest-dir>/genome_atlas_batch.
  --atlas-script ATLAS_SCRIPT
                        Path to genome_quality_atlas.py. Defaults to the copy
                        next to this wrapper.
  --sample-column SAMPLE_COLUMN
                        Column name to add to the masters for the manifest
                        sample. Default: sample
  --category-column CATEGORY_COLUMN
                        Column name to add to the masters for the manifest
                        category. Default: category
  --group-column GROUP_COLUMN
                        Group column passed through to
                        genome_quality_atlas.py. Default: Phylum
  --top-n-groups TOP_N_GROUPS
                        Top-N groups passed through to
                        genome_quality_atlas.py. Default: 12
  --individual-output-name INDIVIDUAL_OUTPUT_NAME
                        Per-directory atlas output folder name. Default:
                        genome_atlas
  --individual-prefix INDIVIDUAL_PREFIX
                        Per-directory atlas file prefix. Default:
                        genome_quality
  --global-prefix GLOBAL_PREFIX
                        Combined atlas file prefix. Default: genome_quality
  --individual-master-name INDIVIDUAL_MASTER_NAME
                        Filename for the augmented per-directory master table.
                        Default: Master_genome_QC.atlas.tsv
  --combined-master-name COMBINED_MASTER_NAME
                        Filename for the combined master table. Default:
                        Master_genome_QC.combined.tsv
  --ani-threshold ANI_THRESHOLD
                        ANI threshold passed through to
                        genome_quality_atlas.py. Default: 95.0
  --ani-af-threshold ANI_AF_THRESHOLD
                        Alignment-fraction threshold passed through to
                        genome_quality_atlas.py. Default: 0.5
  --ani-threads ANI_THREADS
                        FastANI thread count passed through to
                        genome_quality_atlas.py. Default: 1
  --ani-top-overlaps ANI_TOP_OVERLAPS
                        Top overlap combinations passed through to
                        genome_quality_atlas.py. Default: 15
  --ani-results ANI_RESULTS
                        Optional existing FastANI results file to reuse for
                        the global category comparison.
  --matched-samples-only
                        Restrict global category-comparison outputs to samples
                        present in every category. FastANI overlap/dedup is
                        then limited to within-sample matches.
  --best-sample-dir-name BEST_SAMPLE_DIR_NAME
                        Directory name under the combined output for per-
                        sample best deduplicated sets. Default:
                        best_of_sample_sets
  --best-global-dir-name BEST_GLOBAL_DIR_NAME
                        Directory name under the combined output for the
                        global best deduplicated set. Default: best_of_best
  --best-set-min-mimag-tier {low,medium,high}
                        Minimum MIMAG tier allowed into atlas best-of-sample /
                        best-of-best selection. Default: low (keeps all
                        genomes). Use 'medium' to exclude LQ genomes or 'high'
                        to keep only HQ genomes.
  --skip-to-best        Skip manifest processing and atlas generation, and
                        only export best-of-sample / best-of-best sets from
                        existing combined output files.
  --skip-organize-outputs
                        Skip moving outputs into plots/ and tables/
                        subdirectories grouped by style and type.
  --run-best-set-phylogeny
                        After best-of-sample / best-of-best export, run 16S
                        and GTDB-marker phylogeny generation for every best-
                        set directory.
  --best-set-phylogeny-threads BEST_SET_PHYLOGENY_THREADS
                        Threads passed to best_set_phylogeny.py for Barrnap,
                        MAFFT, and GTDB-Tk. Default: 1
  --best-set-gtdbtk-data-path BEST_SET_GTDBTK_DATA_PATH
                        Optional GTDBTK_DATA_PATH override passed to
                        best_set_phylogeny.py when building GTDB marker
                        alignments/trees.
```

## phylogeny

```text
usage: genome_qc phylogeny [-h] [--threads THREADS]
                             [--gtdbtk-data-path GTDBTK_DATA_PATH]
                             [--taxonomy-level {species,genus,none}]
                             root_dir

Build 16S and GTDB-marker phylogenies for selected genome-set directories
under a wrapper output root.

positional arguments:
  root_dir              Wrapper output directory, denovo_phylogeny directory,
                        or a specific selected-set directory.

options:
  -h, --help            show this help message and exit
  --threads THREADS     Threads for MAFFT, Barrnap, and GTDB-Tk. Default: 1
  --gtdbtk-data-path GTDBTK_DATA_PATH
                        Optional GTDBTK_DATA_PATH override for GTDB marker
                        alignments.
  --taxonomy-level {species,genus,none}
                        Minimum resolved taxonomy required before tree
                        construction. Default: species.
```

## render

```text
usage: genome_qc render [-h]
                               {init-config,render,batch,convert-config} ...

Render reproducible, metadata-annotated phylogeny figures with ETE.

positional arguments:
  {init-config,render,batch,convert-config}
    init-config         Write a reusable JSON rendering config.
    render              Render one tree from a JSON config.
    batch               Render every non-empty Newick tree under a root
                        directory.
    convert-config      Convert JSON/YAML render configs while preserving
                        renderer defaults.

options:
  -h, --help            show this help message and exit
```

### render init-config

```text
usage: genome_qc render init-config [-h] [-o OUTPUT] [--tree TREE]
                                           [--metadata [METADATA ...]]
                                           [--output-prefix OUTPUT_PREFIX]

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output JSON config path.
  --tree TREE           Optional tree path to seed autodiscovered
                        metadata/output paths.
  --metadata [METADATA ...]
                        Optional metadata TSV paths.
  --output-prefix OUTPUT_PREFIX
                        Optional output prefix for rendered figures.
```

### render render

```text
usage: genome_qc render render [-h] -c CONFIG [--tree TREE]
                                      [--metadata [METADATA ...]]
                                      [--output-prefix OUTPUT_PREFIX]
                                      [--formats FORMATS [FORMATS ...]]
                                      [--dry-run]

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        JSON rendering config.
  --tree TREE           Override tree path from config.
  --metadata [METADATA ...]
                        Override metadata TSV paths from config.
  --output-prefix OUTPUT_PREFIX
                        Override output prefix from config.
  --formats FORMATS [FORMATS ...]
                        Override output formats, e.g. svg pdf png.
  --dry-run             Print intended output paths without rendering.
```

### render batch

```text
usage: genome_qc render batch [-h] -c CONFIG [-o OUTPUT_DIR]
                                     [--limit LIMIT] [--dry-run]
                                     root

positional arguments:
  root                  Root directory to search for trees.

options:
  -h, --help            show this help message and exit
  -c CONFIG, --config CONFIG
                        Base JSON rendering config.
  -o OUTPUT_DIR, --output-dir OUTPUT_DIR
                        Optional directory for rendered outputs.
  --limit LIMIT         Render only the first N discovered trees.
  --dry-run             Print intended output paths without rendering.
```

### render convert-config

```text
usage: genome_qc render convert-config [-h] -o OUTPUT input

positional arguments:
  input                 Input JSON or YAML config.

options:
  -h, --help            show this help message and exit
  -o OUTPUT, --output OUTPUT
                        Output JSON/YAML config path.
```
