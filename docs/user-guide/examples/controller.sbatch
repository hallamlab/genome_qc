#!/bin/bash
#SBATCH --job-name=genome-qc-controller
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=7-00:00:00
#SBATCH --output=genome-qc-controller-%j.log
set -euo pipefail
# Activate your Genome QC environment before sbatch so its PATH is inherited.
# Supply --account (and, if needed, --partition) to sbatch at submission.
# These JVM settings apply to Nextflow, not to biological tool memory requests.
export NXF_OPTS="${NXF_OPTS:--Xms256m -Xmx4g -XX:ActiveProcessorCount=4}"
exec genome_qc "$@"
