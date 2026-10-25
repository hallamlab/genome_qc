# Results and explorer

## Files to start with

| Path | Contents |
| --- | --- |
| `Genome_inventory.tsv` | Every supplied genome, input provenance and assessment/selection status |
| `Master_genome_QC.tsv` | All assessed representatives, metrics, status columns and selection flags |
| `Selected_genome_QC.tsv` | Subset passing the configured final selection |
| `Assessment_status.tsv` | Tool outcomes, reasons, exit codes and diagnostic log references |
| `genome_quality_atlas/` | Annotated genome table, atlas figures and supported comparison outputs |
| `reports/EDA_portal.html` | Interactive entry point to the result tables and figure gallery |
| `inputs.resolved.json` | Original input paths, IDs and SHA-256 checksums |
| `logs/` | Commands, resource requests, package versions and Nextflow execution reports |

Intermediate tool directories include `seqkit/`, `checkm/`, `qscore/`, `gtdbtk/`,
`barrnap/`, `dedupe/`, `trnascan/` and `gunc/`. The historical name `dedupe` refers
to representative contig-length variants within an input genome. It does not mean
that different genomes were removed as redundant. [Compact mode](recovery.md)
removes disposable intermediates after a fully successful run.

## View a report

```bash
genome_qc report -o ~/genome-qc-output --serve --port 8765
```

Open `http://127.0.0.1:8765`. See [SSH forwarding](quickstart.md#open-the-results)
for remote results. The report command indexes available results without
rerunning the scientific tools; completed compact results reuse their report.

**Summary & atlas** opens first. Genome counts are clickable, allowing direct
inspection of assessed, selected, unselected or incomplete subsets. Figure
previews open enlarged views and PDFs. The atlas describes the original cohort;
interactive table filters do not recalculate figures.

**Explore tables** supports numeric ranges, taxonomy/text filters, sorting,
column selection and CSV export. Numeric endpoints are inclusive; missing values
cannot satisfy an active numeric bound. Criteria combine with AND. Exports
include all matching rows, not just the current page.

**Use matching genomes across tables** carries the current genome set to other
linked tables. Tables without genome links display no rows while this restriction
is active. **Clear genome subset** removes it.

**Select genomes** saves documented subsets; see [selection](selection.md).
The explorer does not change original analysis files.

## Provenance

`logs/<run-id>/run.json` saves the invocation, resolved resource budgets,
Nextflow command and run status. `params.json` records workflow parameters;
`controller.config` records executor settings. `trace.tsv`, `execution.html`
and `timeline.html` describe task execution. `environment-packages.json` records
resolved packages for environments in the selected cache, including versions and
builds. Retain database installation receipts as well as input provenance when
sharing results.
