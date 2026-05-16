#!/bin/bash
# =============================================================================
# OrthoFinder Integration — Within-OG Subfamily Subclustering
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   sbatch orthofinder_cluster.slurm
# Tune the USER CONFIGURATION section below before submitting.
#
# This step replaces the standard MMseqs2 global clustering (Step 1) when
# OrthoFinder HOG or OG sequences are available.  Proteins are pre-grouped
# by their OrthoFinder group membership, then subclustered within each group
# with MMseqs2 at a lower sequence-identity threshold (subcluster_min_seq_id).
# This preserves orthology context while separating recent paralogs.
#
# Input: a directory of *.faa or *.fa files, one per HOG/OG, e.g.:
#   OrthoFinder/Results_*/Phylogenetic_Hierarchical_Orthogroups/N0/
#   OrthoFinder/Results_*/Orthogroup_Sequences/
#
# Optional: --gene-trees-source restricts processing to OGs that have a
# resolved gene tree (OrthoFinder v3 output).  OGs without a tree are skipped.
#
# Key config.yaml options (orthofinder section):
#   subcluster_mode: linclust | cluster | auto
#   subcluster_min_seq_id: 0.4       — lower than global mmseqs for divergent paralogs
#   subcluster_threads: 4            — threads per MMseqs2 OG invocation
#   parallel_og_workers: 16          — OGs processed concurrently
#   Total CPU ≈ parallel_og_workers × subcluster_threads
#
# Discovery partition guidance:
#   epyc-64  — 64 cores (= 16 workers × 4 threads), 248 GB, ≤48 h
#   oneweek  — for very large OrthoFinder runs (>100k OGs) needing >48 h
# =============================================================================
#SBATCH --job-name=plm-orthofinder
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64           # parallel_og_workers × subcluster_threads
#SBATCH --mem=128G                   # larger HOGs can be memory-intensive
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these paths before submitting
# ---------------------------------------------------------------------------
OG_DIR="/path/to/OrthoFinder/Results_*/Orthogroup_Sequences/"
RESULTS_ROOT="/path/to/results"
CONFIG="/path/to/config.yaml"

# Optional: path to gene tree filter for OrthoFinder v3.
# Set to "" to process all OGs.
# Examples:
#   GENE_TREES_SOURCE="/path/to/OrthoFinder/Results_*/Gene_Trees/Resolved_Gene_Trees/"
#   GENE_TREES_SOURCE="/path/to/OrthoFinder/Results_*/Resolved_Gene_Trees.txt"
GENE_TREES_SOURCE=""
# ---------------------------------------------------------------------------

module load conda
conda activate plm_cluster

# Build the --gene-trees-source argument only if a path is provided
GENE_TREES_ARG=()
if [[ -n "${GENE_TREES_SOURCE}" ]]; then
    GENE_TREES_ARG=("--gene-trees-source" "${GENE_TREES_SOURCE}")
fi

plm_cluster orthofinder-cluster \
    --og_dir       "${OG_DIR}" \
    --outdir       "${RESULTS_ROOT}/01_mmseqs" \
    --config       "${CONFIG}" \
    --results_root "${RESULTS_ROOT}" \
    --resume \
    "${GENE_TREES_ARG[@]}"
