#!/bin/bash
#SBATCH -J plm-embed-cpu
#SBATCH -c 16
#SBATCH --mem=64G
#SBATCH -t 24:00:00
plm_cluster embed --reps_fasta $1/01_mmseqs/subfamily_reps.faa --weights_path $2 --config $3 --results_root $1
