#!/bin/bash
# =============================================================================
# Step 5b — Merge HMM-HMM Shard Outputs
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   # Run automatically after the array job completes (recommended):
#   ARRAY_JOB_ID=$(sbatch --parsable hmm_hmm_edges_array.slurm)
#   sbatch --dependency=afterok:${ARRAY_JOB_ID} merge_hmm_shards.slurm
#
#   # Or submit manually once all array tasks have finished:
#   sbatch merge_hmm_shards.slurm
#
# This step concatenates per-shard TSV files produced by
# hmm_hmm_edges_array.slurm into a single merged raw edge table, then
# applies the core/relaxed threshold filters to produce:
#   results/03_hmm_hmm_edges/hmm_hmm_edges_core.tsv
#   results/03_hmm_hmm_edges/hmm_hmm_edges_relaxed.tsv
#
# This step is fast (Python I/O + Polars concat), typically < 5 minutes.
# Discovery partition guidance:
#   main  — sufficient
# =============================================================================
#SBATCH --job-name=plm-hmm-merge
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G                    # increase to 32G if >10M raw edges
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

plm_cluster merge-hmm-shards \
    --outdir       "${RESULTS_ROOT}/03_hmm_hmm_edges" \
    --config       "${CONFIG}" \
    --results_root "${RESULTS_ROOT}" \
    --resume
