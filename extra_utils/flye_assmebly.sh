#!/bin/bash

# this file was used for the CB2A genome announcement
# https://www.ncbi.nlm.nih.gov/nuccore/CP189825.1/
# mamba install -c bioconda flye=2.9.5

flye --genome-size 5m --pacbio-hifi ./reads/CB2A_BD.fastq --out-dir ./assembly_out
