#!/bin/bash
# =============================================================================
# Step 7 — Cluster Families (Leiden / MCL)
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   sbatch cluster_families.slurm
# Tune the USER CONFIGURATION section below before submitting.
#
# This step runs Leiden (default) or MCL graph clustering over the merged
# edges.  It is mostly single-threaded Python and finishes quickly even for
# very large graphs; the main resource requirement is memory to hold the
# edge lists.
#
# Clustering method (--method flag):
#   leiden  — default; uses leidenalg.  Parameters: leiden_resolution_strict
#             and leiden_resolution_functional in config.yaml.
#   mcl     — Markov Cluster Algorithm; requires mcl in PATH.
#
# Discovery partition guidance:
#   main     — default; up to 48 h, adequate for most datasets
#   epyc-64  — if >10 M edges and memory becomes the bottleneck (increase --mem)
# =============================================================================
#SBATCH --job-name=plm-cluster
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=main
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G                    # increase to 64G+ for graphs >10 M edges
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these paths before submitting
# ---------------------------------------------------------------------------
RESULTS_ROOT="/path/to/results"
CONFIG="/path/to/config.yaml"

# Clustering method: "leiden" (default) | "mcl"
CLUSTER_METHOD="leiden"
# ---------------------------------------------------------------------------

module load conda
conda activate plm_cluster

plm_cluster cluster-families \
    --merged_edges_strict      "${RESULTS_ROOT}/06_family_clustering/merged_edges_strict.tsv" \
    --merged_edges_functional  "${RESULTS_ROOT}/06_family_clustering/merged_edges_functional.tsv" \
    --subfamily_map            "${RESULTS_ROOT}/01_mmseqs/subfamily_map.tsv" \
    --outdir                   "${RESULTS_ROOT}/06_family_clustering" \
    --method                   "${CLUSTER_METHOD}" \
    --config                   "${CONFIG}" \
    --results_root             "${RESULTS_ROOT}" \
    --resume
