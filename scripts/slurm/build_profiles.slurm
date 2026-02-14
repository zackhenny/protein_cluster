#!/bin/bash
#SBATCH -J plm-profiles
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 24:00:00
module load hhsuite mafft
plm_cluster build-profiles --proteins_fasta $1 --subfamily_map $3/01_mmseqs/subfamily_map.tsv --config $2 --results_root $3
