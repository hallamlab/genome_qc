# In-house Genome QC Workflow
This is the quality control workflow for genomes, SAGs, MAGs, etc...
It is a Snakemake workflow that relies on an Anaconda environment.

## TLDR; I just want to install and run please!!!

### Prerequistes:
Install your preferred `conda/mamba` manager, `mamba` is suggested for it's speed.
You will need to know where your envs are created for later steps where you save DBs in an env.
To find the path:
```
conda info --base
# OR
mamba info --base
```

### To download:
```
git clone git@github.com:hallamlab/genome_qc.git
cd genome_qc
```

### To install the dependancies:
```
mamba env create -f genome_qc.yml
```

The above `mamba` command will finish with something like this:
```
done
#
# To activate this environment, use
#
#     $ conda activate genome_qc
#
# To deactivate an active environment, use
#
#     $ conda deactivate
```

*NOTE: Several DBs need to be downloaded and installed before the workflow can be fully functional.
Namely, the CheckM, GUNC, GTDB-tk DBs, the latter being quite large.*

### Download and install DBs:
```
# CheckM
# Automatically performed when installed

# GUNC
mkdir "${HOME}/mambaforge/envs/genome_qc/share/gunc_db"
gunc download_db ${HOME}/mambaforge/envs/genome_qc/share/gunc_db

# GTDB-tk
wget --no-check-certificate https://data.gtdb.ecogenomic.org/releases/release220/220.0/auxillary_files/gtdbtk_package/full_package/gtdbtk_r220_data.tar.gz

tar -xvzf gtdbtk_r220_data.tar.gz -C "${HOME}/mambaforge/envs/genome_qc/share/gtdbtk-2.4.0/db/" --strip 1 > /dev/null

rm gtdbtk_r220_data.tar.gz

GTDBTK_DATA_PATH="${HOME}/mambaforge/envs/genome_qc/share/gtdbtk-2.4.0/db/"

```

### Install Snakemake:
```
mamba create -n snakemake snakemake=7.20.0
```

### Configuration
Before you can run the workflow you need to configure the run.  
There is a YAML file named `config_template.yaml` that provides a starting point.  
**First**, open the file and save it as `config.yaml`.  
The contents looks like the following:
```
working_dir: "/full/path/to/working/directory" # global path to WD
fasta_dir: "fasta" # within the WD
nthreads: "1" # <= num CPUs/threads
logs: "logs" # within the WD
condaenv: "genome_qc"
filter_size: "1000" # if empty, no filter. if multiple, space delimited. e.g. "1000 2500 10000"
completeness: "50" # >= checkm completeness
contamination: "5" # <= checkm contamination
qscore_min: "50" # >= derived q-score
gunc_db: "${HOME}/mambaforge/envs/genome_qc/share/gunc_db/gunc_db_progenomes2.1.dmnd" # global path to GUNC database
```

Here is a breakdown of the variable:
- working_dir - This is the GLOBAL path to your projects working dir that contains the `fasta_dir` and will contain all outputs to this workflow.
- fasta_dir - the LOCAL path beneath the working_dir where the fasta files for the genomes can be found, I suggest just naming it `fasta_input` or something obvious.
- nthreads - how many CPUs/threads do you want the workflow to use.
- logs - what would you like to name the logging dir for the pipeline, leave it as `logs` unless you have some reason not to do so.
- condaenv - the name of the conda env, should be `genome_qc`.
- filter_size - if you'd like to prefilter your genomes to 1 or more filtering thresholds.
- completeness - minimum completeness to keep genome, via CheckM.
- contamination - minimum contamination to keep genome, via CheckM.
- qscore_min - minimum q-score to keep genome.
- gunc_db - the GLOBAL path of the GUNC DB.

### Running the Workflow
Once configuration is complete. Run the workflow via snakemake:
```
mamba activate snakemake
snakemake -p -s Snakefile.genome_qc --use-conda --cores all
```

### Test included in Repo
We have included a few genomes in the test directory of this repo.  
Just configure the workflow as above using the test dir and run the workflow.


# Workflow Overview:
This workflow performs high-confidence microbial genome quality control and functional annotation using a reproducible, modular Snakemake pipeline. The pipeline accepts a directory of genome FASTA files and outputs a set of high-quality, deduplicated, annotated genomes with associated quality metrics and metadata. The workflow integrates multiple tools for structural quality assessment, gene prediction, taxonomic classification, contamination detection, and functional annotation.

## Methods: Genome Quality Control Pipeline

### Environment and Execution

This workflow is executed using [Snakemake](https://snakemake.readthedocs.io) and Conda environments to ensure reproducibility and dependency isolation. It is launched with:

```bash
snakemake -p -s Snakefile.genome_qc --use-conda --cores all
```

### Input

The workflow is configured via a `config.yaml` file, which must define:

- `working_dir`: Root output directory.
- `fasta_dir`: Directory containing input genome FASTA files.
- `nthreads`: Number of threads to use.
- `logs`: Subdirectory for log files.
- `filter_size`: Minimum contig size threshold for filtering.
- `completeness`: Minimum CheckM completeness threshold.
- `contamination`: Maximum CheckM contamination threshold.
- `qscore_min`: Minimum composite quality score threshold.
- `condaenv`: Path or name of the Snakemake-managed conda environment.
- `gunc_db`: Path to the GUNC reference database.

### Workflow Steps

#### 1. **Contig Filtering (`run_seqkit_seq`)**
FASTA files are filtered using `seqkit seq` to remove short contigs based on the `filter_size` threshold. Multiple thresholds may be evaluated. Outputs are saved to a filtered FASTA directory, and empty files are removed.

#### 2. **Quality Assessment with CheckM (`run_checkm`)**
Filtered genome bins are evaluated using `CheckM` for genome completeness and contamination. CheckM results are filtered using user-defined thresholds (`completeness` and `contamination`) to select high-quality metagenome-assembled genomes (MAGs). A summary TSV file is produced with these high-quality bins.

#### 3. **Contig Statistics (`run_seqkit_stat`)**
All filtered genome files are profiled using `seqkit stat` to calculate general statistics such as GC content, N50, and total sequence length.

#### 4. **Composite Quality Scoring (`run_qscore`)**
A custom script (`calc_qscore.py`) computes a composite quality score for each genome using completeness, contamination, and assembly statistics. Bins below the `qscore_min` threshold are excluded from downstream analysis.

#### 5. **Subset Genome Extraction (`subset_genomes`)**
Only high-quality genome bins passing all previous filters are copied to a subset directory for downstream analyses.

#### 6. **Taxonomic Classification (`run_gtdbtk`)**
Genome bins are classified taxonomically using `GTDB-Tk` with the `--skip_ani_screen` flag. The tool generates bacterial (`bac120`) and archaeal (`ar53`) summary files for downstream parsing.

#### 7. **rRNA Prediction (`run_barrnap`)**
The `barrnap` tool is used to detect ribosomal RNA genes in bacterial and archaeal genomes separately. If the corresponding GTDB-Tk summary file is not present, the step is skipped gracefully.

#### 8. **Self-BLAST of rRNAs (`run_blast`)**
Each rRNA FASTA file is BLASTed against itself to detect redundancy or contamination. BLAST databases are constructed for each genome, and top hits are saved.

#### 9. **rRNA Comparison (`compile_barblast`)**
A custom Python script (`compare_barrnap.py`) processes `barrnap` and BLAST outputs to identify inconsistencies or duplicated rRNA elements.

#### 10. **Redundancy Removal (`run_dedupe`)**
The `dedupe_genomes.py` script uses quality scores and rRNA comparisons to identify and remove redundant genomes. A filtered list of non-redundant, high-quality bins is saved.

#### 11. **Copying Final Genomes (`run_cp_dedupe`)**
Deduplicated, high-quality genome FASTA files are copied into a new directory for final analysis.

#### 12. **tRNA Annotation (`run_trnscan`)**
`tRNAscan-SE` is run in parallel (per genome) on the final set of bins. The pipeline distinguishes between Bacteria and Archaea based on the deduplication table and uses appropriate flags (`-B` or `-A`). If no genomes of a kingdom are present, that kingdom's step is skipped safely. All results are parsed using `trnascan_parse.py`.

#### 13. **Contamination Screening with GUNC (`run_gunc`)**
`GUNC` is run on the deduplicated genomes using a specified reference database to detect taxonomic inconsistencies and chimeric bins.

#### 14. **Final Metadata Merge (`run_merge_master`)**
The `merge_master.py` script integrates results from GUNC, tRNA annotation, GTDB-Tk, CheckM, and deduplication into a master metadata file summarizing quality, taxonomy, and function for each genome.

### Output Directory Structure Overview

```
test/
├── Master_genome_QC.tsv
├── Master_genome_QC_*.pdf/png         # Summary plots (PDF and PNG)
├── barrnap/                           # rRNA annotations and summary tables
├── checkm/                            # CheckM output tables and logs
├── dedupe/                            # Deduplication tables and final FASTA files
├── fasta/                             # Original input genomes
├── fasta_filtered/                    # Filtered genomes by contig length
├── fasta_subset/                      # High-quality filtered genomes used in downstream steps
├── gtdbtk/                            # GTDB-Tk taxonomic classification outputs
├── gunc/                              # GUNC contamination screen and DIAMOND results
├── logs/                              # Snakemake logs for each rule
├── qscore/                            # Composite quality scores (raw and high-quality)
├── seqkit/                            # Genome statistics generated by SeqKit
└── trnascan/                          # tRNA annotations (GFF, BED, FASTA, logs, stats)
```

---

### 📖 File and Directory Descriptions

| Path                         | Description |
|------------------------------|-------------|
| `Master_genome_QC.tsv`       | Final merged metadata table summarizing genome quality, taxonomy, and annotations. |
| `Master_genome_QC_*.pdf/png` | Visualizations of genome QC metrics, stratified by taxonomy or score threshold. |
| `barrnap/`                   | rRNA gene predictions and BLAST comparisons. Includes `.fasta`, `.gff`, and summary TSVs. |
| `checkm/`                    | Outputs from CheckM including completeness/contamination assessments. |
| `dedupe/`                    | Genomes selected after removing redundancy based on QC and rRNA. Includes final `fasta/` directory. |
| `fasta/`                     | Raw input genome files in compressed FASTA format. |
| `fasta_filtered/`            | Genomes after contig-length filtering using SeqKit. |
| `fasta_subset/`              | High-quality genome subset used for downstream steps. `.fai` files are sequence indexes. |
| `gtdbtk/`                    | Taxonomic classification results using GTDB-Tk, including summary tables and marker alignments. |
| `gunc/`                      | GUNC outputs identifying chimeric genomes and contamination using gene calls and DIAMOND hits. |
| `logs/`                      | Logs from each Snakemake rule, useful for debugging and provenance. |
| `qscore/`                    | Quality score tables including all bins and filtered high-quality ones (`mqhp`). |
| `seqkit/`                    | Contig statistics for all genomes including GC%, length, and N50. |
| `trnascan/`                  | tRNA predictions per genome, including GFF, FASTA, BED, logs, and summary TSVs. |


### Reproducibility Notes

- All software is managed through Snakemake’s `--use-conda` support.
- Each rule runs in an isolated conda environment defined by the workflow.
- The workflow is deterministic and can resume from failed steps.