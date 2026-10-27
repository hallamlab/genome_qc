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

## Check browser-rendered diagrams

A successful Sphinx build does not establish that Mermaid diagrams render.
The documentation CI also opens the built guide in Chromium and checks for real
flowchart nodes and connectors, with no Mermaid error graphics or browser errors.
To run the same check locally after building:

```bash
python -m pip install playwright==1.55.0
python -m playwright install --with-deps chromium
python scripts/check_docs_browser.py /tmp/genome-qc-guide
```

This requires access to the Mermaid CDN used by the guide. A network failure is
reported as a failed check, not a successful diagram render.

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

## Keep reference pages current

`python scripts/build_cli_reference.py` regenerates help for every public
subcommand, including nested render commands. Run it in a Python environment
with NumPy, pandas, SciPy and statsmodels available for the standalone parsers;
it invokes only `--help` and does not start analyses. Ordinary Sphinx/RTD builds
use the checked-in reference and do not need those additional imports.

When output schemas change, update [the column dictionary](output-columns.md)
from the producing code, including units, missing-value behavior, formulas and
status enums. Check named table headers from a completed validation run against
the dictionary. Treat headerless upstream files separately rather than mistaking
their first data row for column names. Optional metadata columns retain the
user-supplied definitions. Do not copy validation datasets or study tables into
the documentation tree.
