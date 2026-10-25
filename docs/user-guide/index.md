# Genome QC

Genome QC brings the evidence needed to assess a genome into one place. Start
with bacterial or archaeal isolate genomes, single-cell amplified genomes (SAGs),
or metagenome-assembled genomes (MAGs). The workflow checks assembly structure,
completeness, contamination, taxonomy, and RNA genes, then brings those results
into a genome atlas and an interactive explorer.

You do not have to decide which genomes to keep before assessing them. By default,
every usable genome continues through the applicable tools. The final table keeps
available measurements and explicitly identifies failed or missing assessments.
Use the explorer to compare genomes, select a subset, and save both the selected
table and a reproducible record of your criteria.

**[Quickstart](quickstart.md)** · **[Test your installation](test.md)** ·
**[Issues and feature requests](https://github.com/hallamlab/genome_qc/issues)**

```{image} assets/workflow-brief.svg
:alt: Genome QC nodal overview, from genome FASTAs through independent assessments to the atlas and explorer.
:align: center
:class: workflow-brief
```

[Full-size SVG](assets/workflow-brief.svg) · [PDF](assets/workflow-brief.pdf) ·
[Detailed workflow](workflow.md)

The same controller supports a local Linux server and a Slurm cluster. Nextflow
manages dependencies, parallel jobs, and resumable tasks; Mamba supplies isolated
tool environments. Missing databases are prepared automatically for a full run.
Read [resources](resources.md) before launching on a shared server or cluster.

```{toctree}
:maxdepth: 2
:caption: Start here

installation
quickstart
test
inputs
```

```{toctree}
:maxdepth: 2
:caption: Run and explore

workflow
commands
databases
resources
outputs
selection
recovery
```

```{toctree}
:maxdepth: 1
:caption: Reference

interpretation
cli-reference
citations
development
```
