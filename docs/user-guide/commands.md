# Commands

See the generated [CLI reference](cli-reference.md) for every controller flag.

The complete `run` also builds [cohort ANI groups and annotated marker trees](phylogeny.md)
by default. These use the shared resource controls and need no study labels.
The standalone `phylogeny` command below serves the separate selected-set comparison workflow.

## Nextflow-backed stages

These commands accept the shared [execution options](resources.md). Use a separate
output directory per standalone stage, and supply genome FASTAs or a manifest.
A full `run` wires these stages together automatically.

| Command | Purpose | Required input |
| --- | --- | --- |
| `run` | Complete genome assessment, final selection, atlas and explorer | `-i` or `--manifest` |
| `build_db` | Install/reuse CheckM, GTDB R226 and GUNC references | `-o`, optional `-d` |
| `preprocess` | Contig-length variants and assembly statistics | Genome FASTAs |
| `quality` | CheckM assessment of supplied genomes | Genome FASTAs |
| `taxonomy` | GTDB-Tk classification | Genome FASTAs |
| `rrna` | barrnap prediction and rRNA self-comparison | Genome FASTAs; `--kingdom` if needed |
| `trna` | tRNAscan-SE prediction | Genome FASTAs; `--kingdom` if needed |
| `chimera` | GUNC screening | Genome FASTAs |

For example:

```bash
genome_qc taxonomy -i ~/genomes -o ~/genome-taxonomy \
  -d ~/genome-qc-databases --threads 8 --max_cpus 32
genome_qc rrna -i ~/genomes -o ~/genome-rrna \
  --kingdom Bacteria --threads 8 --max_cpus 32
```

Standalone stages assess the files supplied to them; they do not automatically
consume a previous command's output or build the full atlas. Use `run` for the
connected workflow. `--plan` validates inputs and writes parameters/commands
without launching Nextflow, but does create the output directory and stage input
files. Use a disposable planning directory, not an active run's directory.

## Standalone analyses

| Command | Purpose |
| --- | --- |
| `atlas` | Build an atlas from a master QC table |
| `compare` | Combine completed runs and compare labelled genome sets |
| `phylogeny` | Build trees for selected sets produced by the comparison workflow |
| `render` | Render Newick trees with ETE using a reusable JSON style |
| `report` | Index output tables and serve the explorer |
| `check` | Locate controller prerequisites |

These commands run locally in the invoking environment. They do not create a
Slurm job or automatically provision missing standalone-analysis tools. The
complete `run` supplies the atlas environment automatically. For standalone
atlas/comparison work with FastANI:

```bash
mamba env create -f envs/genome_quality_atlas.yaml
conda activate genome_qc_genome_quality_atlas
python -m pip install --no-deps .
genome_qc atlas ~/genome-qc-output/Master_genome_QC.tsv -o ~/atlas-refresh
```

The environment commands above are run from the repository checkout. The optional
phylogeny/render tools have additional dependencies; inspect `phylogeny --help`,
`render --help` and `envs/ete_render.yaml` before using them.

A comparison manifest is tab-separated:

```text
sample	category	run_directory
sample_1	method_A	/shared/results/method_A
sample_1	method_B	/shared/results/method_B
```

```bash
genome_qc compare ~/comparison.tsv -o ~/genome-comparison
genome_qc atlas --help
genome_qc compare --help
genome_qc phylogeny --help
genome_qc render --help
```

The standalone analyses expose their own full argument help. Do not infer
statistical replication from a directory name: comparison labels and sample
relationships must match the study design. See [interpretation](interpretation.md).
