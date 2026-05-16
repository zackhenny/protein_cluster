#!/bin/bash
# =============================================================================
# Step 6 — Merge Graphs (Strict + Functional)
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   sbatch merge_graph.slurm
# Tune the USER CONFIGURATION section below before submitting.
#
# This step combines the HMM-HMM edges (core and relaxed) with the embedding
# KNN edges into two merged edge tables:
#   merged_edges_strict.tsv     — HMM core edges only
#   merged_edges_functional.tsv — HMM relaxed + embedding edges, merged by policy
#
# Edge weight policy (graph.edge_weight_policy in config.yaml):
#   strict               — HMM core edges only (ignores embeddings)
#   union                — max(hmm_weight, embedding_weight) per pair
#   gated                — embedding edges only where HMM weight >= weak_hmm_threshold
#   downweight_embeddings — w_hmm*hmm + w_emb*emb  (default; recommended)
#
# This is a fast Python + Polars step; typical runtime < 10 minutes.
# Discovery partition guidance:
#   main  — sufficient for most datasets
# =============================================================================
#SBATCH --job-name=plm-merge-graph
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                    # increase to 32G for very large edge sets
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these paths before submitting
# ---------------------------------------------------------------------------
RESULTS_ROOT="/path/to/results"
CONFIG="/path/to/config.yaml"
# ---------------------------------------------------------------------------

module load conda
conda activate plm_cluster

plm_cluster merge-graph \
    --hmm_core       "${RESULTS_ROOT}/03_hmm_hmm_edges/hmm_hmm_edges_core.tsv" \
    --hmm_relaxed    "${RESULTS_ROOT}/03_hmm_hmm_edges/hmm_hmm_edges_relaxed.tsv" \
    --embedding_edges "${RESULTS_ROOT}/04_embeddings/embedding_knn_edges.tsv" \
    --outdir         "${RESULTS_ROOT}/06_family_clustering" \
    --config         "${CONFIG}" \
    --results_root   "${RESULTS_ROOT}" \
    --resume
