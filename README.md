# In-house Genome QC Workflow
This is the quality control workflow for genomes, SAGs, MAGs, etc...
It is a Snakemake workflow that relies on an Anaconda environment.

## COMING SOON:
- glyph visualization
- file manifest

## Prerequistes:
Install your preferred `conda/mamba` manager, `mamba` is suggested for it's speed.
You will need to know where your envs are created for later steps where you save DBs in an env.
To find the path:
```
conda info --base
# OR
mamba info --base
```

## To download:
```
git clone git@github.com:hallamlab/genome_qc.git
cd genome_qc
```

## To install the dependancies:
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
Namely, the CheckM, GUNC, GTDB-tk DBs, the later being quite large.*

## Download and install DBs:
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

## Install Snakemake:
```
mamba create -n snakemake snakemake=7.20.0
```

## Configuration
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

## Running the Workflow
Once configuration is complete. Run the workflow via snakemake:
```
mamba activate snakemake
snakemake -p -s Snakefile.genome_qc --use-conda --cores all
```

## Test included in Repo
We have included a few genomes in the test directory of this repo.  
Just configure the workflow as above using the test dir and run the workflow.
