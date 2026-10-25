# CLI reference

Generated from the controller parser by `scripts/build_cli_reference.py`.
For standalone atlas/comparison/tree commands, use their individual `--help`;
see [commands](commands.md). Resource defaults resolved at runtime are explained
in [resources](resources.md).

```text
usage: genome_qc [-h] [--version]
                 {run,build_db,preprocess,quality,taxonomy,rrna,trna,chimera,atlas,compare,phylogeny,render,check,report} ...

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
usage: genome_qc run [-h] (-i INPUT | --manifest MANIFEST) -o OUTPUT
                     [-d DATABASE] [--threads THREADS] [--max_cpus MAX_CPUS]
                     [--max_memory MAX_MEMORY] [--memory MEMORY]
                     [--checkm_memory CHECKM_MEMORY]
                     [--gtdb_memory GTDB_MEMORY] [--gunc_memory GUNC_MEMORY]
                     [--trnascan_memory TRNASCAN_MEMORY]
                     [--atlas_memory ATLAS_MEMORY] [--max_tasks MAX_TASKS]
                     [--executor {local,slurm}] [--account ACCOUNT]
                     [--partition PARTITION] [--qos QOS]
                     [--reservation RESERVATION] [--time_limit TIME_LIMIT]
                     [--submit_rate SUBMIT_RATE] [--work_dir WORK_DIR]
                     [--conda_cache CONDA_CACHE] [--keep_work]
                     [--compact_results] [--scratch_dir SCRATCH_DIR]
                     [--resume] [--plan] [--params PARAMS]
                     [--max_batch_size MAX_BATCH_SIZE]
                     [--filter_size FILTER_SIZE]
                     [--min_genome_size MIN_GENOME_SIZE]
                     [--max_genome_size MAX_GENOME_SIZE]
                     [--completeness COMPLETENESS]
                     [--contamination CONTAMINATION] [--qscore_min QSCORE_MIN]
                     [--require_rrna_consistency]
                     [--ani_threshold ANI_THRESHOLD]

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Genome FASTA directory (searched recursively)
  --manifest MANIFEST   TSV with id and fasta; relative paths resolve beside
                        the manifest
  -o, --output OUTPUT
  -d, --database DATABASE
  --threads THREADS
  --resume
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
                        physical/cgroup capacity, not free RAM at startup
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
                        ANI threshold (%); default: 99
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
                          [--resume] [--plan] [--params PARAMS]

options:
  -h, --help            show this help message and exit
  -o, --output OUTPUT
  -d, --database DATABASE
  --threads THREADS
  --resume
  --plan                Validate inputs and save commands without executing
  --params PARAMS       Advanced Nextflow parameters as a JSON object

Execution and resources:
  --max_cpus MAX_CPUS   Total CPU budget; local defaults to available CPUs,
                        Slurm has no default aggregate cap
  --max_memory MAX_MEMORY
                        Total memory budget; local defaults to 90% of
                        physical/cgroup capacity, not free RAM at startup
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
usage: genome_qc preprocess [-h] (-i INPUT | --manifest MANIFEST) -o OUTPUT
                            [-d DATABASE] [--threads THREADS]
                            [--max_cpus MAX_CPUS] [--max_memory MAX_MEMORY]
                            [--memory MEMORY] [--max_tasks MAX_TASKS]
                            [--executor {local,slurm}] [--account ACCOUNT]
                            [--partition PARTITION] [--qos QOS]
                            [--reservation RESERVATION]
                            [--time_limit TIME_LIMIT]
                            [--submit_rate SUBMIT_RATE] [--work_dir WORK_DIR]
                            [--conda_cache CONDA_CACHE] [--keep_work]
                            [--compact_results] [--scratch_dir SCRATCH_DIR]
                            [--resume] [--plan] [--params PARAMS]
                            [--filter_size FILTER_SIZE]

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Genome FASTA directory (searched recursively)
  --manifest MANIFEST   TSV with id and fasta; relative paths resolve beside
                        the manifest
  -o, --output OUTPUT
  -d, --database DATABASE
  --threads THREADS
  --resume
  --plan                Validate inputs and save commands without executing
  --params PARAMS       Advanced Nextflow parameters as a JSON object

Execution and resources:
  --max_cpus MAX_CPUS   Total CPU budget; local defaults to available CPUs,
                        Slurm has no default aggregate cap
  --max_memory MAX_MEMORY
                        Total memory budget; local defaults to 90% of
                        physical/cgroup capacity, not free RAM at startup
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
usage: genome_qc quality [-h] (-i INPUT | --manifest MANIFEST) -o OUTPUT
                         [-d DATABASE] [--threads THREADS]
                         [--max_cpus MAX_CPUS] [--max_memory MAX_MEMORY]
                         [--memory MEMORY] [--checkm_memory CHECKM_MEMORY]
                         [--max_tasks MAX_TASKS] [--executor {local,slurm}]
                         [--account ACCOUNT] [--partition PARTITION]
                         [--qos QOS] [--reservation RESERVATION]
                         [--time_limit TIME_LIMIT] [--submit_rate SUBMIT_RATE]
                         [--work_dir WORK_DIR] [--conda_cache CONDA_CACHE]
                         [--keep_work] [--compact_results]
                         [--scratch_dir SCRATCH_DIR] [--resume] [--plan]
                         [--params PARAMS] [--max_batch_size MAX_BATCH_SIZE]

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Genome FASTA directory (searched recursively)
  --manifest MANIFEST   TSV with id and fasta; relative paths resolve beside
                        the manifest
  -o, --output OUTPUT
  -d, --database DATABASE
  --threads THREADS
  --resume
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
                        physical/cgroup capacity, not free RAM at startup
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
usage: genome_qc taxonomy [-h] (-i INPUT | --manifest MANIFEST) -o OUTPUT
                          [-d DATABASE] [--threads THREADS]
                          [--max_cpus MAX_CPUS] [--max_memory MAX_MEMORY]
                          [--memory MEMORY] [--gtdb_memory GTDB_MEMORY]
                          [--max_tasks MAX_TASKS] [--executor {local,slurm}]
                          [--account ACCOUNT] [--partition PARTITION]
                          [--qos QOS] [--reservation RESERVATION]
                          [--time_limit TIME_LIMIT]
                          [--submit_rate SUBMIT_RATE] [--work_dir WORK_DIR]
                          [--conda_cache CONDA_CACHE] [--keep_work]
                          [--compact_results] [--scratch_dir SCRATCH_DIR]
                          [--resume] [--plan] [--params PARAMS]
                          [--max_batch_size MAX_BATCH_SIZE]

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Genome FASTA directory (searched recursively)
  --manifest MANIFEST   TSV with id and fasta; relative paths resolve beside
                        the manifest
  -o, --output OUTPUT
  -d, --database DATABASE
  --threads THREADS
  --resume
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
                        physical/cgroup capacity, not free RAM at startup
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
usage: genome_qc rrna [-h] (-i INPUT | --manifest MANIFEST) -o OUTPUT
                      [-d DATABASE] [--threads THREADS] [--max_cpus MAX_CPUS]
                      [--max_memory MAX_MEMORY] [--memory MEMORY]
                      [--max_tasks MAX_TASKS] [--executor {local,slurm}]
                      [--account ACCOUNT] [--partition PARTITION] [--qos QOS]
                      [--reservation RESERVATION] [--time_limit TIME_LIMIT]
                      [--submit_rate SUBMIT_RATE] [--work_dir WORK_DIR]
                      [--conda_cache CONDA_CACHE] [--keep_work]
                      [--compact_results] [--scratch_dir SCRATCH_DIR]
                      [--resume] [--plan] [--params PARAMS]
                      [--kingdom {Bacteria,Archaea,Eukaryota}]

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Genome FASTA directory (searched recursively)
  --manifest MANIFEST   TSV with id and fasta; relative paths resolve beside
                        the manifest
  -o, --output OUTPUT
  -d, --database DATABASE
  --threads THREADS
  --resume
  --plan                Validate inputs and save commands without executing
  --params PARAMS       Advanced Nextflow parameters as a JSON object
  --kingdom {Bacteria,Archaea,Eukaryota}

Execution and resources:
  --max_cpus MAX_CPUS   Total CPU budget; local defaults to available CPUs,
                        Slurm has no default aggregate cap
  --max_memory MAX_MEMORY
                        Total memory budget; local defaults to 90% of
                        physical/cgroup capacity, not free RAM at startup
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
usage: genome_qc trna [-h] (-i INPUT | --manifest MANIFEST) -o OUTPUT
                      [-d DATABASE] [--threads THREADS] [--max_cpus MAX_CPUS]
                      [--max_memory MAX_MEMORY] [--memory MEMORY]
                      [--trnascan_memory TRNASCAN_MEMORY]
                      [--max_tasks MAX_TASKS] [--executor {local,slurm}]
                      [--account ACCOUNT] [--partition PARTITION] [--qos QOS]
                      [--reservation RESERVATION] [--time_limit TIME_LIMIT]
                      [--submit_rate SUBMIT_RATE] [--work_dir WORK_DIR]
                      [--conda_cache CONDA_CACHE] [--keep_work]
                      [--compact_results] [--scratch_dir SCRATCH_DIR]
                      [--resume] [--plan] [--params PARAMS]
                      [--kingdom {Bacteria,Archaea,Eukaryota}]

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Genome FASTA directory (searched recursively)
  --manifest MANIFEST   TSV with id and fasta; relative paths resolve beside
                        the manifest
  -o, --output OUTPUT
  -d, --database DATABASE
  --threads THREADS
  --resume
  --plan                Validate inputs and save commands without executing
  --params PARAMS       Advanced Nextflow parameters as a JSON object
  --kingdom {Bacteria,Archaea,Eukaryota}

Execution and resources:
  --max_cpus MAX_CPUS   Total CPU budget; local defaults to available CPUs,
                        Slurm has no default aggregate cap
  --max_memory MAX_MEMORY
                        Total memory budget; local defaults to 90% of
                        physical/cgroup capacity, not free RAM at startup
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
usage: genome_qc chimera [-h] (-i INPUT | --manifest MANIFEST) -o OUTPUT
                         [-d DATABASE] [--threads THREADS]
                         [--max_cpus MAX_CPUS] [--max_memory MAX_MEMORY]
                         [--memory MEMORY] [--gunc_memory GUNC_MEMORY]
                         [--max_tasks MAX_TASKS] [--executor {local,slurm}]
                         [--account ACCOUNT] [--partition PARTITION]
                         [--qos QOS] [--reservation RESERVATION]
                         [--time_limit TIME_LIMIT] [--submit_rate SUBMIT_RATE]
                         [--work_dir WORK_DIR] [--conda_cache CONDA_CACHE]
                         [--keep_work] [--compact_results]
                         [--scratch_dir SCRATCH_DIR] [--resume] [--plan]
                         [--params PARAMS] [--max_batch_size MAX_BATCH_SIZE]

options:
  -h, --help            show this help message and exit
  -i, --input INPUT     Genome FASTA directory (searched recursively)
  --manifest MANIFEST   TSV with id and fasta; relative paths resolve beside
                        the manifest
  -o, --output OUTPUT
  -d, --database DATABASE
  --threads THREADS
  --resume
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
                        physical/cgroup capacity, not free RAM at startup
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
  -h, --help           show this help message and exit
  -o, --output OUTPUT
  --serve
  --port PORT
```

## check

```text
usage: genome_qc check [-h]

options:
  -h, --help  show this help message and exit
```
