#!/bin/bash
# =============================================================================
# Step 8b — Write Membership Matrices
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   sbatch write_matrices.slurm
# Tune the USER CONFIGURATION section below before submitting.
#
# This step converts the protein-family segment table into sparse membership
# matrices in two formats (TSV and optionally Matrix Market .mtx):
#   results/07_membership_matrices/protein_family_strict.tsv
#   results/07_membership_matrices/protein_family_functional.tsv
#   results/07_membership_matrices/*.mtx  (if outputs.write_matrix_market: true)
#
# Key config.yaml options (outputs section):
#   write_dense_threshold: 2000000   — max cells before skipping dense output
#   write_matrix_market: true        — also write .mtx sparse format
#
# This step is fast Python + Polars I/O; typical runtime < 5 minutes.
# Discovery partition guidance:
#   main  — sufficient for all dataset sizes
# =============================================================================
#SBATCH --job-name=plm-matrices
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G                    # increase to 64G for very large protein sets
#SBATCH --time=02:00:00
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

plm_cluster write-matrices \
    --subfamily_map          "${RESULTS_ROOT}/01_mmseqs/subfamily_map.tsv" \
    --protein_family_segments "${RESULTS_ROOT}/05_domain_hits/protein_family_segments.tsv" \
    --outdir                 "${RESULTS_ROOT}/07_membership_matrices" \
    --config                 "${CONFIG}" \
    --results_root           "${RESULTS_ROOT}" \
    --resume
