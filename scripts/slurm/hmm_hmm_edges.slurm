#!/bin/bash
#SBATCH -J plm-hhm
#SBATCH -c 32
#SBATCH --mem=64G
#SBATCH -t 24:00:00
module load hhsuite
plm_cluster hmm-hmm-edges \
  --profile_index $2/02_profiles/subfamily_profile_index.tsv \
  --candidate_edges $2/04_embeddings/embedding_knn_edges.tsv \
  --config $1 --results_root $2
