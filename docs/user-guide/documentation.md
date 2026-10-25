# Documentation

The [README](https://github.com/hallamlab/genome_qc#readme) provides the installation,
quickstart, test entry point and brief workflow. This guide holds the complete
instructions, organized in the same progression as MetaPathways, ASPIRE and SCARAB:

- **Getting started:** installation, first run, installation test and inputs.
- **Analysis guides:** individual commands, databases, resources and recovery.
- **Results and reference:** explorer, selection, interpretation, workflows,
  command-line reference and citations.
- **Development:** packaging, contributions and documentation maintenance.

## Build the guide

From a repository checkout:

```bash
mamba create -n genome-qc-docs -c conda-forge python=3.11 pip
conda activate genome-qc-docs
python -m pip install -r docs/user-guide/requirements.txt
python scripts/check_workflow_style.py
python -m sphinx -W --keep-going -b html docs/user-guide /tmp/genome-qc-guide
```

Open `/tmp/genome-qc-guide/index.html`. Documentation builds do not require the
biological tools or reference databases.

## Keep the layout consistent

Use short introductions, complete commands, and links from the first run to
advanced settings. Keep installation methods on the installation page and
scientific caveats with interpretation. The brief nodal workflow belongs on the
README and landing page; expanded workflow and architecture diagrams belong on
the workflow page. Follow `docs/WORKFLOW_STYLE.md` when editing figures.

Only `docs/user-guide/` is public documentation. The Sphinx page allow-list in
`conf.py` deliberately excludes adjacent local manuscripts and research results.
Add new public pages to that list and to the appropriate landing-page navigation
group. Do not include study-specific manuscripts, tables or figures.
