#!/bin/bash
# =============================================================================
# Step 3 — ESM-2 Embeddings (GPU)
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   sbatch embed_gpu.slurm
# Tune the USER CONFIGURATION section below before submitting.
#
# GPU embedding is 10–50× faster than CPU for typical datasets.
# Set embed.device: cuda in config.yaml (or it will fall back to CPU).
#
# Discovery GPU partition guidance:
#   gpu  — all GPU nodes; use --constraint to target specific hardware:
#     --constraint=a100   → epyc-7513 + 2× A100 40 GB   (best for large batches)
#     --constraint=a40    → epyc-7313/7282 + 2× A40 48 GB
#     --constraint=l40s   → epyc-9534 + 3× L40S 48 GB   (newest, fastest)
#     --constraint=v100   → xeon-6130 + 2× V100 32 GB
#   Remove --constraint to let Slurm pick any available GPU node.
#
# Memory: the 650M ESM-2 model needs ~3 GB GPU VRAM; a batch_size of 16
# fits comfortably on any of the above GPUs.
# Increase embed.batch_size in config.yaml to 16–32 when using A100/L40S.
# =============================================================================
#SBATCH --job-name=plm-embed-gpu
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # CPU threads for data loading
#SBATCH --gres=gpu:a100:1            # request 1 A100; change to gpu:a40:1 etc.
#SBATCH --mem=64G
#SBATCH --time=08:00:00              # GPU jobs finish much faster than CPU
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these paths before submitting
# ---------------------------------------------------------------------------
RESULTS_ROOT="/path/to/results"
WEIGHTS_PATH="/path/to/esm2_t33_650M_UR50D.pt"   # local ESM-2 model weights
CONFIG="/path/to/config.yaml"
# Make sure embed.device is set to "cuda" in your config.yaml, e.g.:
#   embed:
#     device: cuda
#     batch_size: 16      # increase from default 4 for GPU
# ---------------------------------------------------------------------------

module load conda
conda activate plm_cluster

plm_cluster embed \
    --reps_fasta   "${RESULTS_ROOT}/01_mmseqs/subfamily_reps.faa" \
    --weights_path "${WEIGHTS_PATH}" \
    --outdir       "${RESULTS_ROOT}/04_embeddings" \
    --config       "${CONFIG}" \
    --results_root "${RESULTS_ROOT}" \
    --resume
