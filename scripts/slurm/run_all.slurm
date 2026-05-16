#!/bin/bash
# =============================================================================
# Full Pipeline — run-all (single-node, small-to-medium datasets)
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   sbatch run_all.slurm
# Tune the USER CONFIGURATION section below before submitting.
#
# This script runs all 9 pipeline steps in sequence on a single node:
#   1. MMseqs2 subfamily clustering
#   2. Build subfamily profiles (MAFFT + HH-suite)
#   3. ESM-2 embeddings (CPU)
#   4. KNN candidate generation
#   5. HMM-HMM profile-profile edges
#   6. Merge graphs (strict + functional)
#   7. Leiden family clustering
#   8a. Map proteins to families
#   8b. Write membership matrices
#   9. QC plots
#
# When to use this script vs separate per-step scripts:
#   run_all.slurm     — suitable when the full pipeline fits in ≤48 h
#                       on a single node (~5k–20k subfamilies, CPU embed)
#   Separate scripts  — required when any single step (especially embed or
#                       HMM-HMM) needs GPU, or when HMM-HMM benefits from
#                       multi-node sharding (see hmm_hmm_edges_array.slurm)
#
# HMM mode options (--hmm-mode flag or hmm_hmm.mode in config.yaml):
#   pairwise      — one hhalign per pair (default, requires hhalign)
#   db-search     — hhsearch against ffindex DB (requires hhsearch etc.)
#   mmseqs-profile — MMseqs2 all-vs-all; fastest for >5k subfamilies
#
# KNN mode options (--knn-mode flag or knn.mode in config.yaml):
#   knn    — standard FAISS cosine KNN (default)
#   rkcnn  — Random K-Conditional Nearest Neighbor (slower, higher recall)
#
# Discovery partition guidance:
#   epyc-64  — 64 cores, 248 GB, ≤48 h  (recommended for CPU-only runs)
#   oneweek  — up to 7 days; use for large datasets or slow embed on CPU
#
# For GPU embedding, run embed_gpu.slurm first, then re-submit run-all with
# --resume to skip the already-completed embed step.
# =============================================================================
#SBATCH --job-name=plm-run-all
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64           # used by mmseqs, build-profiles, hmm-hmm
#SBATCH --mem=248G                   # full node memory for epyc-64
#SBATCH --time=2-00:00:00            # 2 days; increase to 5-00:00:00 for large runs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these paths before submitting
# ---------------------------------------------------------------------------
PROTEINS_FASTA="/path/to/proteins.faa"
WEIGHTS_PATH="/path/to/esm2_t33_650M_UR50D.pt"
CONFIG="/path/to/config.yaml"
RESULTS_ROOT="/path/to/results"

# HMM mode: "pairwise" | "db-search" | "mmseqs-profile"
# Recommendation: use "mmseqs-profile" for >5k subfamilies
HMM_MODE="pairwise"

# KNN mode: "knn" | "rkcnn"
KNN_MODE="knn"

MMSEQS_TMPDIR="${TMPDIR:-/tmp}/mmseqs_runall_$$"
mkdir -p "$MMSEQS_TMPDIR"
# ---------------------------------------------------------------------------

module load conda
conda activate plm_cluster
# For pairwise / db-search HMM mode, also load HH-suite:
#   module load hhsuite/3.3.0

plm_cluster run-all \
    --proteins_fasta "${PROTEINS_FASTA}" \
    --weights_path   "${WEIGHTS_PATH}" \
    --config         "${CONFIG}" \
    --results_root   "${RESULTS_ROOT}" \
    --hmm-mode       "${HMM_MODE}" \
    --knn-mode       "${KNN_MODE}" \
    --resume

rm -rf "$MMSEQS_TMPDIR"
