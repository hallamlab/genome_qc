# Development and distribution

## Documentation

The public guide is isolated in `docs/user-guide/`. Manuscripts and study-specific
results elsewhere under `docs/` are not documentation build inputs and must not
be added to public releases. Read the Docs uses the repository-root
`.readthedocs.yaml` and builds only the guide. The project can be linked to this
GitHub repository once these files are pushed; its hostname is not assumed here.

Refresh the command reference with `python scripts/build_cli_reference.py` when
the parser changes. Build locally from the repository:

```bash
python -m venv /tmp/genome-qc-docs
/tmp/genome-qc-docs/bin/python -m pip install -r docs/user-guide/requirements.txt
/tmp/genome-qc-docs/bin/python -m sphinx -W --keep-going -b html \
  docs/user-guide /tmp/genome-qc-docs-html
```

Workflow figures follow `docs/WORKFLOW_STYLE.md`. Rebuild SVG/PDF exports and
inspect both levels of detail after changes:

```bash
/tmp/genome-qc-docs/bin/python -m pip install -r docs/user-guide/diagram-requirements.txt
/tmp/genome-qc-docs/bin/python scripts/build_workflow_figures.py
python scripts/check_workflow_style.py
```

## Tests

```bash
python -m unittest discover -s tests -v
```

Nextflow integration tests are opt-in and require Nextflow, Java, SeqKit, BLAST+
and the atlas Python dependencies on `PATH`. They use explicitly simulated
responses for the major assessment tools; they do not download biological databases.
Use a development environment, not an environment serving an active run:

```bash
GQC_NF_INTEGRATION=1 python -m unittest discover \
  -s tests -p test_workflow_integration.py -v
```

The integration suite checks normal execution, failure continuation/recovery,
compact scratch handling, deterministic bounded batches and cached resume.
A real-genome run remains necessary to validate external-tool interoperability.

## Distribution plan

The primary release route is an **Anaconda-hosted Conda package installed
with Mamba**. The recipe packages the controller, workflow assets and environment
definitions; tool environments remain isolated and large databases external.
Users install through Mamba without a Git checkout or pip step. Package upload
remains separate from the local build and validation described below.

A Quay image should follow validation of that same package. Decide whether it
contains prebuilt tool environments or provisions them into a writable external
cache; a controller-only image is not automatically an offline full workflow.
Validate Docker and Apptainer separately. In particular, a container launching
Slurm needs access to scheduler clients, shared paths and the environment cache;
a local container success does not establish HPC support.

Before publication:

1. Validate the ongoing full-workflow real-genome test, including incomplete
   assessments, atlas, explorer, balanced batching and compact output.
2. Build a Conda artifact from the intended revision, install it in a clean
   environment, and run CLI, packaged-asset and integration checks.
3. Provide a small distributable real-genome test with provenance and reuse terms.
4. Validate the image against the same test; record versions and image digest.
5. Publish only through explicit release jobs after tests pass. Keep Zenodo
   deposits manual and check that no study/manuscript files enter artifacts.

The Anaconda organization/channel, package name, release version, Quay repository
and RTD hostname should be agreed before activation. Publishing credentials belong
in the repository's secret store, never in documentation or source files.
See [Conda build documentation](https://docs.conda.io/projects/conda-build/en/latest/)
and [Quay repository permissions](https://docs.quay.io/guides/repo-permissions.html).

## Building the Conda package

From the repository, create isolated build tools and build the artifact:

```bash
mamba create -n genome-qc-build --override-channels -c conda-forge \
  python=3.11 conda-build conda-index pyyaml
conda activate genome-qc-build
python scripts/prepare_conda_recipe.py --output dist/conda-source
conda build dist/conda-source/recipe --override-channels \
  -c conda-forge -c bioconda --no-anaconda-upload --output-folder dist/channel
conda index dist/channel
```

The preparation command requires a new output directory. It creates a checksummed
archive from an explicit source allow-list, excluding manuscripts, local analyses,
databases and caches. It records the revision, working-tree state and file hashes.
Run dependencies come from `envs/controller.yaml`; pip is a build dependency.
The recipe tests installed commands, packaged workflow assets, planning and report
creation outside the checkout.

Install from the resulting local channel using Mamba, without pip:

```bash
mamba create -n genome-qc-package-test --override-channels --strict-channel-priority \
  -c "file://$PWD/dist/channel" -c conda-forge -c bioconda genome-qc
conda activate genome-qc-package-test
genome_qc check
```

`.github/workflows/conda.yml` builds and validates the artifact, installs it into a
clean environment and runs synthetic Nextflow integration tests against its
installed workflow assets. Public upload is not triggered by a normal build.
The ongoing real-genome assessment remains a separate biological interoperability
test; mocked tool responses do not replace it.
