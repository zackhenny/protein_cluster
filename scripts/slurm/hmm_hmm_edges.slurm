#!/bin/bash
# =============================================================================
# Step 5 — HMM-HMM Profile-Profile Edges (pairwise hhalign, single node)
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   sbatch hmm_hmm_edges.slurm
# Tune the USER CONFIGURATION section below before submitting.
#
# This script runs the pairwise hhalign mode on a single node.  It is well-
# suited for small-to-medium datasets (up to ~5,000 subfamilies / ~1M pairs).
# For larger datasets see:
#   hmm_hmm_edges_array.slurm  — sharded job array (pairwise or db-search)
#   hmm_hmm_edges_mmseqs.slurm — MMseqs2 profile-profile (fastest, 220k+ fams)
#
# Mode selection (--mode flag or hmm_hmm.mode in config.yaml):
#   pairwise    — one hhalign per candidate pair; parallel_workers concurrent
#                 jobs.  Requires hhalign in PATH.  Default.
#   db-search   — build HH-suite ffindex DB + one hhsearch per query;
#                 better for dense candidate sets.  Requires hhsearch,
#                 ffindex_build, cstranslate.
#   mmseqs-profile — see hmm_hmm_edges_mmseqs.slurm
#
# Discovery partition guidance:
#   epyc-64  — 64 cores, 248 GB, ≤48 h  (recommended)
#   oneweek  — same nodes, ≤7 days       (for very large datasets)
#
# Set hmm_hmm.parallel_workers in config.yaml equal to --cpus-per-task.
# =============================================================================
#SBATCH --job-name=plm-hhm
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32           # match hmm_hmm.parallel_workers in config.yaml
#SBATCH --mem=64G
#SBATCH --time=48:00:00              # use oneweek partition for jobs >48 h
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these paths before submitting
# ---------------------------------------------------------------------------
RESULTS_ROOT="/path/to/results"
CONFIG="/path/to/config.yaml"

# Execution mode: "pairwise" (default) | "db-search" | "mmseqs-profile"
# Can also be set via hmm_hmm.mode in config.yaml; CLI flag takes precedence.
HMM_MODE="pairwise"
# ---------------------------------------------------------------------------

module load conda
conda activate plm_cluster
# Alternative: load HH-suite as an environment module:
#   module load hhsuite/3.3.0

# --resume allows restarting after preemption; completed pairs are skipped
# via the NDJSON progress log in results/03_hmm_hmm_edges/.
plm_cluster hmm-hmm-edges \
    --profile_index  "${RESULTS_ROOT}/02_profiles/subfamily_profile_index.tsv" \
    --candidate_edges "${RESULTS_ROOT}/04_embeddings/embedding_knn_edges.tsv" \
    --outdir         "${RESULTS_ROOT}/03_hmm_hmm_edges" \
    --config         "${CONFIG}" \
    --results_root   "${RESULTS_ROOT}" \
    --mode           "${HMM_MODE}" \
    --resume
