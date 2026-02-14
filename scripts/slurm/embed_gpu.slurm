#!/bin/bash
#SBATCH -J plm-embed-gpu
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
plm_cluster embed --reps_fasta $1/01_mmseqs/subfamily_reps.faa --weights_path $2 --config $3 --results_root $1
