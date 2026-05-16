#!/bin/bash
# =============================================================================
# Step 4a — KNN Candidate Neighbor Generation (GPU)
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   sbatch knn_gpu.slurm
# Tune the USER CONFIGURATION section below before submitting.
#
# GPU-accelerated KNN uses FAISS-GPU for the neighbor index, providing a
# significant speedup for large embedding sets (>5,000 subfamily reps).
# Requires the faiss-gpu package; the pipeline falls back to CPU FAISS or
# sklearn if faiss-gpu is not installed.
#
# GPU is also beneficial for rKCNN mode (knn.mode: rkcnn), which uses
# PyTorch tensors for pairwise distance computation when knn.device: cuda.
#
# Ensure your config.yaml has:
#   knn:
#     device: cuda    # enables FAISS-GPU
#     mode: knn       # or "rkcnn" for supervised ensemble
#
# Discovery GPU partition guidance:
#   gpu — all GPU nodes.  KNN is not very GPU-memory intensive; any GPU works.
#     --constraint=a40  or --constraint=a100 are good defaults.
#     --constraint=l40s for newest L40S nodes.
#
# =============================================================================
#SBATCH --job-name=plm-knn-gpu
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:a40:1             # 1 GPU; change to gpu:a100:1 / gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=01:00:00              # GPU KNN is fast; rKCNN may need 2–4 h
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
# Make sure knn.device is set to "cuda" in config.yaml
KNN_MODE="knn"
# ---------------------------------------------------------------------------

module load conda
conda activate plm_cluster

plm_cluster knn \
    --embeddings    "${RESULTS_ROOT}/04_embeddings/embeddings.npy" \
    --ids           "${RESULTS_ROOT}/04_embeddings/ids.txt" \
    --lengths       "${RESULTS_ROOT}/04_embeddings/lengths.tsv" \
    --subfamily_map "${RESULTS_ROOT}/01_mmseqs/subfamily_map.tsv" \
    --out_tsv       "${RESULTS_ROOT}/04_embeddings/embedding_knn_edges.tsv" \
    --mode          "${KNN_MODE}" \
    --config        "${CONFIG}" \
    --results_root  "${RESULTS_ROOT}" \
    --resume
