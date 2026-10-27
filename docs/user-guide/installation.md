# Installation

## Choose an installation route

| Your situation | Route |
| --- | --- |
| You want a normal installed command | Use the Mamba package |
| You are developing or testing source changes | Use the source-install commands below |
| You already have a working installation | Activate its environment; do not reinstall for every run |

Commands in this guide are entered in a Linux terminal, not in Python or R.
An environment is a separate collection of software packages. `mamba` creates or
updates it; `conda activate genome_qc` makes its commands available in the current
terminal. If neither command is available, install
[Miniforge](https://github.com/conda-forge/miniforge), or use your institution's
supported Conda/Mamba installation, and open a new terminal after shell setup.

Genome QC currently targets 64-bit Linux. These instructions are not a tested
native Windows or macOS installation route. A remote Linux server is suitable;
[SSH forwarding](quickstart.md#open-the-results) lets you view its report locally.

## Mamba installation

Genome QC is packaged as `genome-qc`; its command is `genome_qc` (with
`genome-qc` also available as an alias). Install it with:

```bash
mamba create -n genome_qc --override-channels --strict-channel-priority \
  -c conda-forge -c bioconda -c hallamlab genome-qc
conda activate genome_qc
genome_qc --version
genome_qc check
```

Version 0.1.0 is published on the normal `hallamlab` Anaconda channel for
64-bit Linux. The full R232 end-to-end test, package integration tests and
public-channel installation passed. No special label, checkout, Git dependency
install or separate pip step is needed.

Use 64-bit Linux with Mamba available on `PATH`.
[Miniforge](https://github.com/conda-forge/miniforge) supplies Conda and Mamba.
The package installs the CLI, Python libraries, Nextflow, Java, Mamba, workflow
assets and tool environment definitions. Nextflow prepares biological tool
environments automatically when needed. Separate environments keep the different
CheckM and GTDB-Tk Python requirements compatible. Large reference databases are
not part of the package; a full run prepares missing databases at the path set
with `-d` and reuses complete installations.

`check` locates controller prerequisites. It does not validate database contents
or run the biological tools. Continue with [the test](test.md) or
[quickstart](quickstart.md). Even a small genome set needs substantial reference
storage and memory; see [resources](resources.md).

## Updating a package installation

Finish or stop an active analysis before updating the installed software:

```bash
mamba update -n genome_qc --override-channels --strict-channel-priority \
  -c conda-forge -c bioconda -c hallamlab genome-qc
```

For reproducibility, request a specific published version with
`genome-qc=VERSION` when creating an environment. Save the package and reference
provenance with your results. Use a new output directory when scientific settings
change; [resume rules](recovery.md) distinguish interrupted and completed compact runs.

## Source installation for development

This is the developer route, separate from normal package installation:

```bash
git clone https://github.com/hallamlab/genome_qc.git
cd genome_qc
mamba env create -n genome_qc -f envs/controller.yaml
conda activate genome_qc
python -m pip install --no-deps .
```

The environment pins Nextflow 26.10.3; the workflow requires at least that version.
The same controller dependencies feed the Conda recipe. See
[package development](development.md#building-the-conda-package) to build and
validate a local artifact. Docker/Apptainer distribution remains a separate
validation and publication step.
