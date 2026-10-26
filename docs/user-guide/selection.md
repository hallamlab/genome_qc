# Selecting genomes after assessment

Genome QC separates measurement from selection. By default, every assessed genome
is retained, even if a measurement is unavailable. Acceptance thresholds never
skip biological assessment tools. GUNC and tRNA metrics are reported, but are not
automatic acceptance gates.

## Final command-line thresholds

| Flag | Inclusive acceptance rule | Default |
| --- | --- | --- |
| `--min_genome_size` | Genome length at least this many bp | Unrestricted |
| `--max_genome_size` | Genome length no more than this many bp | Unrestricted |
| `--completeness` | CheckM completeness at least this percentage | Unrestricted |
| `--contamination` | CheckM contamination no more than this percentage | Unrestricted |
| `--qscore_min` | Qscore at least this value | Unrestricted |
| `--require_rrna_consistency` | rRNA consistency check must pass | Disabled |

An enabled criterion excludes missing measurements for that criterion. The master
records `pass_genome_size`, `pass_completeness`, `pass_contamination`,
`pass_qscore`, `pass_rRNA_check`, `selected` and `selection_reasons`.
These flags retain the entire master table; the selected subset has its own TSV.
`--filter_size` is different: it changes the contigs being measured.

### Apply thresholds to a new run

For example, to select genomes with completeness at least 90% and contamination
at most 5% while still assessing and retaining measurements for every genome:

```bash
genome_qc run -i ~/genomes -o ~/genome-qc-output-selected \
  -d ~/genome-qc-databases --threads 8 --max_cpus 32 \
  --completeness 90 --contamination 5
```

These are example criteria, not recommended cutoffs for every study. Do not
change thresholds and reuse an old output directory expecting a fresh analysis.
For completed results, the explorer provides the simpler route: define and
export a new subset without rerunning assessment tools or changing the master.

## Reproducible selection in the explorer

Open **Select genomes** to start with the complete master table and no filters.
Add quality ranges, taxonomy matches or other column criteria. Review the number
and identities of matching genomes before exporting:

1. Download the selected CSV and genome-ID list.
2. Download the JSON selection record alongside them.
3. Reload that record to reproduce the same selection against the same indexed data.

The record saves criteria, selected IDs, software version, timestamp and a SHA-256
fingerprint of the indexed source table. Reloading verifies both the fingerprint
and membership. It will not silently apply a selection to changed data. Criteria
combine with AND; numeric ranges include their endpoints and reject missing
values. Table selection is read-only and does not regenerate atlas figures.

`selected=true` is not a guarantee of complete assessment: with unrestricted
criteria, a genome can be selected while a tool failed. Inspect
`assessment_complete` and tool-status columns when defining a final biological set.
