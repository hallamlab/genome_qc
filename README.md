# Genome QC

Genome QC assesses bacterial and archaeal genomes, SAGs and MAGs from FASTA files
and brings their assembly quality, completeness, contamination, taxonomy and RNA
gene measurements into a genome atlas and interactive explorer. Every usable
genome receives the applicable assessments before final selection; missing or
failed measurements remain explicit.

**[Full user guide](docs/user-guide/index.md)** · **[Test](docs/user-guide/test.md)** ·
**[Issues and feature requests](https://github.com/hallamlab/genome_qc/issues)**

## Quickstart

Install the Conda package with Mamba on 64-bit Linux:

```bash
mamba create -n genome_qc --override-channels --strict-channel-priority \
  -c conda-forge -c bioconda -c hallamlab genome-qc
conda activate genome_qc
genome_qc check
```

**Publication status:** the package recipe and validation workflow are prepared;
the HallamLab upload is pending. The command above is the public installation
route once published. No Git checkout or pip step is needed for that route.
See [installation](docs/user-guide/installation.md) for package and developer details.

Put your genome FASTAs in `~/genomes`, then run:

```bash
genome_qc run -i ~/genomes -o ~/genome-qc-output \
  -d ~/genome-qc-databases --threads 8 --max_cpus 32
genome_qc report -o ~/genome-qc-output --serve --port 8765
```

Missing databases and tool environments are prepared automatically. Existing
complete references are reused. Read the [resource guide](docs/user-guide/resources.md)
before your first full run: even a small genome set needs substantial reference
storage and memory. Open `http://127.0.0.1:8765` to inspect the results, filter
linked tables and export a documented genome selection.

For a first test, follow the [small-set walkthrough](docs/user-guide/test.md).
Use `--manifest` for explicit genome IDs, `--executor slurm` for a cluster,
`--resume` after interruption, and `--compact_results` for smaller completed
outputs. See [commands](docs/user-guide/commands.md),
[inputs](docs/user-guide/inputs.md) and [compact mode](docs/user-guide/recovery.md).

## Workflow

<p align="center"><a href="docs/user-guide/workflow.md"><img src="docs/user-guide/assets/workflow-brief.svg?v=connected-nodal-20261025" width="620" alt="Genome QC nodal workflow"></a></p>

[SVG](docs/user-guide/assets/workflow-brief.svg) ·
[PDF](docs/user-guide/assets/workflow-brief.pdf) ·
[Detailed workflow](docs/user-guide/workflow.md)

The full guide is ready for Read the Docs. Until the project URL is configured,
the links above point to the guide in this repository. The Conda package is the primary installation route; Quay publication is a
separate release step, described in the [distribution plan](docs/user-guide/development.md#distribution-plan).

Please cite the [tools and references used](docs/user-guide/citations.md) and
retain the run's input, database and software provenance with your results.
