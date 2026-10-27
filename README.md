# Genome QC

Genome QC assesses bacterial and archaeal genomes, SAGs and MAGs from FASTA files
and brings their assembly quality, completeness, contamination, taxonomy and RNA
gene measurements into a genome atlas and interactive explorer. Full runs also
include ANI groups and annotated GTDB-marker trees. Every usable
genome receives the applicable assessments before final selection; missing or
failed measurements remain explicit.

**[Full user guide](https://hallamlab-genome-qc.readthedocs.io/en/latest/)** · **[Test](https://hallamlab-genome-qc.readthedocs.io/en/latest/test.html)** ·
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

## Check the installation

Verify the installed controller without downloading databases or running genome assessments:

```bash
conda activate genome_qc
genome_qc --version
genome_qc check
genome_qc run --help
```

`check` verifies that Nextflow, Java and Mamba are available on `PATH`. It does
not validate reference databases or biological tool execution. This is the
recommended initial setup check. A full biological run, even with only a few
genomes, can take hours because GTDB must load and place sequences against large
reference trees; it is not a quick installation test.

For optional full-workflow validation and interpreting results, see the
[test guide](https://hallamlab-genome-qc.readthedocs.io/en/latest/test.html).
For manifests, Slurm, resuming and compact outputs, see the
**[full user guide](https://hallamlab-genome-qc.readthedocs.io/en/latest/)**.

## Workflow

<p align="center"><a href="docs/user-guide/workflow.md"><img src="docs/user-guide/assets/workflow-brief.svg?v=connected-nodal-20261025" width="620" alt="Genome QC nodal workflow"></a></p>

[SVG](docs/user-guide/assets/workflow-brief.svg) ·
[PDF](docs/user-guide/assets/workflow-brief.pdf) ·
[Detailed workflow](docs/user-guide/workflow.md)

## Support and citation

[Report issues or request features](https://github.com/hallamlab/genome_qc/issues).
Please cite the [tools and references used](docs/user-guide/citations.md) and
retain the run's input, database and software provenance with your results.
Documentation source is maintained in `docs/user-guide/`.
