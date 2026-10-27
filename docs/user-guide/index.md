# Genome QC

Assess bacterial and archaeal genomes, compare assembly quality, completeness,
contamination, taxonomy and RNA genes, and explore the results in a genome atlas.
Run on a local server or submit work through Slurm using the same commands.

**Start here:** [install Genome QC](installation.md), then [test your installation](test.md).
Follow the [quickstart](quickstart.md) to run your own genome FASTAs from assessment
to the explorer. A full run prepares missing databases automatically; check the
[storage and memory requirements](resources.md) before starting.

```{image} assets/workflow-brief.svg
:alt: Genome QC nodal overview, from genome FASTAs through independent assessments to the atlas and explorer.
:align: center
:class: workflow-brief
```

[Full-size SVG](assets/workflow-brief.svg) · [PDF](assets/workflow-brief.pdf) ·
[Detailed workflow](workflow.md)

All usable genomes receive the applicable assessments before final selection.
The master table retains available measurements and records failed or missing
assessments. Use the explorer to compare genomes and export a subset together with
a reproducible record of your criteria. Blue circles mark inputs, diamonds mark
processes, green circles mark outputs, and black dots mark branch junctions.

## Start simple

```{toctree}
:maxdepth: 1
:caption: Getting started

installation
quickstart
test
inputs
```

## Run your analysis

```{toctree}
:maxdepth: 1
:caption: Analysis guides

commands
databases
resources
recovery
```

## Understand and explore

```{toctree}
:maxdepth: 1
:caption: Results and reference

outputs
output-columns
selection
phylogeny
interpretation
workflow
cli-reference
citations
```

## Maintain and contribute

```{toctree}
:maxdepth: 1
:caption: Development

development
documentation
```

[Source code](https://github.com/hallamlab/genome_qc) ·
[Issues and feature requests](https://github.com/hallamlab/genome_qc/issues) ·
[Tool citations](citations.md)
