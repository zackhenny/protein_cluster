#!/bin/bash
# =============================================================================
# Step 4a — KNN Candidate Neighbor Generation (CPU)
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   sbatch knn.slurm
# Tune the USER CONFIGURATION section below before submitting.
#
# This step generates candidate subfamily pairs from ESM-2 embeddings using
# FAISS (fast approximate nearest neighbor).  It must run after the embed step.
#
# Two modes are available (--mode flag or knn.mode in config.yaml):
#   knn    — standard FAISS / sklearn cosine KNN (default; fast, unsupervised)
#   rkcnn  — Random K-Conditional Nearest Neighbor; supervised ensemble using
#            MMseqs2 subfamily labels.  Better recall for remote homologs at
#            the cost of higher CPU time.  Requires --subfamily_map.
#            See docs/algorithm_background.md for details.
#
# For GPU-accelerated KNN / rKCNN see knn_gpu.slurm.
#
# Discovery partition guidance:
#   main  — sufficient; KNN over <50k embeddings finishes in minutes
#   epyc-64 — for rKCNN with many subspaces (rkcnn_n_subspaces >= 50)
#
# Memory: 8–16 GB is adequate for <50k embeddings (1280-D ESM-2 vectors).
# Increase to 32 GB for 100k+ embeddings.
# =============================================================================
#SBATCH --job-name=plm-knn
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G                    # increase to 32G for 100k+ embeddings
#SBATCH --time=02:00:00              # rKCNN can take longer; increase to 8h if needed
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these paths before submitting
# ---------------------------------------------------------------------------
RESULTS_ROOT="/path/to/results"
CONFIG="/path/to/config.yaml"

# KNN mode: "knn" (default) | "rkcnn"
KNN_MODE="knn"
# ---------------------------------------------------------------------------

module load conda
conda activate plm_cluster

plm_cluster knn \
    --embeddings   "${RESULTS_ROOT}/04_embeddings/embeddings.npy" \
    --ids          "${RESULTS_ROOT}/04_embeddings/ids.txt" \
    --lengths      "${RESULTS_ROOT}/04_embeddings/lengths.tsv" \
    --subfamily_map "${RESULTS_ROOT}/01_mmseqs/subfamily_map.tsv" \
    --out_tsv      "${RESULTS_ROOT}/04_embeddings/embedding_knn_edges.tsv" \
    --mode         "${KNN_MODE}" \
    --config       "${CONFIG}" \
    --results_root "${RESULTS_ROOT}" \
    --resume
