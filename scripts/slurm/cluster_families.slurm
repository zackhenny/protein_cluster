#!/bin/bash
#SBATCH -J plm-cluster
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -t 04:00:00
plm_cluster cluster-families --merged_edges $1/06_family_clustering/merged_edges.tsv --subfamily_map $1/01_mmseqs/subfamily_map.tsv --config $2 --results_root $1
