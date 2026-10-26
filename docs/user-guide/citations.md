# Software and citations

Report the Genome QC version, source revision, input checksums, reference releases
and resolved tool versions for the run. Environment specifications are not lockfiles:
several dependencies are unpinned. Use `logs/*/environment-packages.json` and the
tool logs for the versions actually used, rather than assuming one version from
this guide. No Genome QC paper or DOI is asserted here.

| Component | Role | Primary reference or project |
| --- | --- | --- |
| Nextflow | Workflow execution and caching | [Project and documentation](https://www.nextflow.io/) |
| Mamba | Isolated dependency environments | [Project](https://github.com/mamba-org/mamba) |
| SeqKit | FASTA preprocessing and assembly statistics | [Project](https://github.com/shenwei356/seqkit) |
| CheckM | Completeness, contamination and strain heterogeneity | Parks et al. (2015), [DOI: 10.1101/gr.186072.114](https://doi.org/10.1101/gr.186072.114); [project](https://github.com/Ecogenomics/CheckM) |
| GTDB-Tk | Genome taxonomy; this workflow uses 2.8.0 with R232 | Chaumeil et al. (2022), [DOI: 10.1093/bioinformatics/btac672](https://doi.org/10.1093/bioinformatics/btac672); [documentation](https://ecogenomics.github.io/GTDBTk/) |
| GTDB | Taxonomic reference release | [Database](https://gtdb.ecogenomic.org/) |
| barrnap | rRNA gene prediction | [Project](https://github.com/tseemann/barrnap) |
| BLAST+ | rRNA self-comparison | [NCBI BLAST](https://blast.ncbi.nlm.nih.gov/Blast.cgi) |
| tRNAscan-SE | tRNA gene prediction | Chan et al. (2021), [DOI: 10.1093/nar/gkab688](https://doi.org/10.1093/nar/gkab688); [project](https://github.com/UCSC-LoweLab/tRNAscan-SE) |
| GUNC | Gene-based chimerism/contamination assessment | Orakov et al. (2021), [DOI: 10.1186/s13059-021-02393-0](https://doi.org/10.1186/s13059-021-02393-0); [project](https://github.com/grp-bork/gunc) |
| FastANI | Configured ANI comparisons in the atlas | [Project](https://github.com/ParBLiSS/FastANI) |
| pandas, NumPy, SciPy | Tables and numerical calculations | [pandas](https://pandas.pydata.org/), [NumPy](https://numpy.org/), [SciPy](https://scipy.org/) |
| statsmodels | Configured comparison models | [Project](https://www.statsmodels.org/) |
| Matplotlib, seaborn | Atlas figures | [Matplotlib](https://matplotlib.org/), [seaborn](https://seaborn.pydata.org/) |
| ETE | Optional tree rendering | [Project](https://etetoolkit.org/) |

Tool-internal dependencies, including placement, gene prediction and sequence
search programs, are also recorded in the resolved environment manifests. Cite
those tools when reporting analyses that use them. Database receipts record
sources and checksums separately from package versions.
