#!/bin/bash
# =============================================================================
# Step 5 — HMM-HMM Edges: Sharded Job Array (pairwise or db-search)
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   # Submit the array (16 shards):
#   ARRAY_JOB_ID=$(sbatch --parsable hmm_hmm_edges_array.slurm)
#
#   # Submit the merge step, to run only after ALL array tasks succeed:
#   sbatch --dependency=afterok:${ARRAY_JOB_ID} merge_hmm_shards.slurm
#
# Tune N_SHARDS and the SBATCH directives below before submitting.
#
# Why sharding?
# The candidate pair list is split into N independent chunks.  Each Slurm
# array task processes one chunk, writing its own shard TSV and progress
# NDJSON.  This is the recommended approach for large datasets (>5,000
# subfamilies) where a single job would exceed the 48-hour partition limit
# on Discovery's epyc-64 / main partitions.
#
# Choosing N_SHARDS:
#   - Aim for each shard to finish in 8–12 hours on 32 cores.
#   - Rule of thumb: N_SHARDS ≈ total_candidate_pairs / 200_000
#   - For 1M pairs → 8 shards;  for 5M pairs → 32 shards
#   - Also update --array=0-<N_SHARDS-1> to match N_SHARDS.
#
# Mode (pairwise vs db-search):
#   pairwise   — one hhalign per pair; good default.  Requires hhalign.
#   db-search  — builds HH-suite ffindex DB + runs hhsearch per query;
#                faster for dense candidate sets.  Requires hhsearch,
#                ffindex_build, cstranslate.
#
# --resume: each task can be re-submitted independently if preempted.
# The per-shard NDJSON log tracks completed pairs so work is never repeated.
#
# Discovery partition guidance:
#   epyc-64  — 64 cores, 248 GB, ≤48 h per task (recommended)
#   main     — acceptable; slightly lower core count on some nodes
#   oneweek  — if individual shards might exceed 48 h
# =============================================================================
#SBATCH --job-name=plm-hmm-arr
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32           # match hmm_hmm.parallel_workers in config.yaml
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --array=0-15                 # 16 shards (0-indexed); adjust to match N_SHARDS-1
#SBATCH --output=logs/%x_%A_%a.out  # %A = job array ID, %a = task index
#SBATCH --error=logs/%x_%A_%a.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these before submitting
# ---------------------------------------------------------------------------
RESULTS_ROOT="/path/to/results"
CONFIG="/path/to/config.yaml"

# Must match the --array upper bound + 1 (e.g. --array=0-15 → N_SHARDS=16)
N_SHARDS=16

# Execution mode for this array: "pairwise" | "db-search"
# Use hmm_hmm_edges_mmseqs.slurm for "mmseqs-profile" mode instead.
HMM_MODE="pairwise"
# ---------------------------------------------------------------------------

module load conda
conda activate plm_cluster
# Alternative for db-search mode (needs hhsearch, ffindex_build, cstranslate):
#   module load hhsuite/3.3.0

# SLURM_ARRAY_TASK_ID is automatically set to the shard index (0, 1, 2, ...)
plm_cluster hmm-hmm-edges \
    --profile_index   "${RESULTS_ROOT}/02_profiles/subfamily_profile_index.tsv" \
    --candidate_edges "${RESULTS_ROOT}/04_embeddings/embedding_knn_edges.tsv" \
    --outdir          "${RESULTS_ROOT}/03_hmm_hmm_edges" \
    --config          "${CONFIG}" \
    --results_root    "${RESULTS_ROOT}" \
    --mode            "${HMM_MODE}" \
    --shard-id        "${SLURM_ARRAY_TASK_ID}" \
    --n-shards        "${N_SHARDS}" \
    --resume
