#!/bin/bash
#SBATCH -J plm-mmseqs
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 12:00:00
module load mmseqs2
plm_cluster mmseqs-cluster --proteins_fasta $1 --config $2 --results_root $3
