#!/bin/bash
# =============================================================================
# Full Pipeline — run-all-orthofinder (single-node, OrthoFinder input)
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   sbatch run_all_orthofinder.slurm
# Tune the USER CONFIGURATION section below before submitting.
#
# This script runs the full pipeline using OrthoFinder HOG/OG sequences
# instead of a global proteins FASTA.  Step 1 (orthofinder-cluster) replaces
# the standard MMseqs2 clustering; all subsequent steps are identical to
# run-all.
#
# Steps:
#   1.  OrthoFinder within-OG subclustering (replaces mmseqs-cluster)
#   2.  Build subfamily profiles (MAFFT + HH-suite)
#   3.  ESM-2 embeddings
#   4.  KNN candidate generation
#   5.  HMM-HMM profile-profile edges
#   6.  Merge graphs
#   7.  Leiden family clustering
#   8a. Map proteins to families
#   8b. Write membership matrices
#   9.  QC plots
#
# Input: OG_DIR — directory of *.faa / *.fa files, one per OG/HOG, e.g.:
#   OrthoFinder/Results_*/Orthogroup_Sequences/
#   OrthoFinder/Results_*/Phylogenetic_Hierarchical_Orthogroups/N0/
#
# Optional: GENE_TREES_SOURCE restricts to OGs with a resolved gene tree.
# See orthofinder_cluster.slurm for details on this filter.
#
# Discovery partition guidance:
#   epyc-64  — 64 cores, 248 GB, ≤48 h
#   oneweek  — for very large OrthoFinder outputs needing >48 h
# =============================================================================
#SBATCH --job-name=plm-run-all-of
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=248G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these paths before submitting
# ---------------------------------------------------------------------------
OG_DIR="/path/to/OrthoFinder/Results_*/Orthogroup_Sequences/"
WEIGHTS_PATH="/path/to/esm2_t33_650M_UR50D.pt"
CONFIG="/path/to/config.yaml"
RESULTS_ROOT="/path/to/results"

# HMM mode: "pairwise" | "db-search" | "mmseqs-profile"
HMM_MODE="pairwise"

# KNN mode: "knn" | "rkcnn"
KNN_MODE="knn"

# Gene trees filter (OrthoFinder v3).  Set to "" to process all OGs.
GENE_TREES_SOURCE=""

MMSEQS_TMPDIR="${TMPDIR:-/tmp}/mmseqs_of_$$"
mkdir -p "$MMSEQS_TMPDIR"
# ---------------------------------------------------------------------------

module load conda
conda activate plm_cluster

# Build the --gene-trees-source argument only if a path is provided
GENE_TREES_ARG=()
if [[ -n "${GENE_TREES_SOURCE}" ]]; then
    GENE_TREES_ARG=("--gene-trees-source" "${GENE_TREES_SOURCE}")
fi

plm_cluster run-all-orthofinder \
    --og_dir         "${OG_DIR}" \
    --weights_path   "${WEIGHTS_PATH}" \
    --config         "${CONFIG}" \
    --results_root   "${RESULTS_ROOT}" \
    --hmm-mode       "${HMM_MODE}" \
    --knn-mode       "${KNN_MODE}" \
    --resume \
    "${GENE_TREES_ARG[@]}"

rm -rf "$MMSEQS_TMPDIR"
