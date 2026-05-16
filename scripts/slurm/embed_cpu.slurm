#!/bin/bash
# =============================================================================
# Step 3 — ESM-2 Embeddings (CPU)
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   sbatch embed_cpu.slurm
# Tune the USER CONFIGURATION section below before submitting.
#
# CPU embedding is slow but requires no GPU allocation.  For datasets with
# more than ~5,000 subfamily representatives the GPU variant (embed_gpu.slurm)
# is strongly recommended — it is typically 10–50× faster.
#
# Discovery partition guidance:
#   main     — default; up to 48 h, Xeon/EPYC nodes
#   oneweek  — for very large datasets needing >48 h on CPU
#
# embed.long_seq_policy in config.yaml controls handling of long sequences:
#   truncate (default) — cap at embed.max_len residues (fast, recommended)
#   skip               — omit sequences longer than max_len
#   full               — process full-length (slow, high memory for >3k aa)
# =============================================================================
#SBATCH --job-name=plm-embed-cpu
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8            # ESM-2 uses PyTorch; 8 threads is a good default
#SBATCH --mem=64G                    # increase to 128G if processing very long sequences
#SBATCH --time=48:00:00              # CPU embedding is slow; use oneweek for >10k reps
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

# Optional: store per-batch checkpoints so a preempted job can resume.
# Set to "" to disable; recommended for CPU runs which are long.
CHECKPOINT_DIR="${RESULTS_ROOT}/04_embeddings/ckpt"
# ---------------------------------------------------------------------------

module load conda
conda activate plm_cluster

# If checkpointing is enabled, pass the dir via config or set embed.checkpoint_dir
# in config.yaml.  The --resume flag below will reload completed batches.

plm_cluster embed \
    --reps_fasta   "${RESULTS_ROOT}/01_mmseqs/subfamily_reps.faa" \
    --weights_path "${WEIGHTS_PATH}" \
    --outdir       "${RESULTS_ROOT}/04_embeddings" \
    --config       "${CONFIG}" \
    --results_root "${RESULTS_ROOT}" \
    --resume
