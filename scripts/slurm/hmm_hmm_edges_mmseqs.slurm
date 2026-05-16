#!/bin/bash
# =============================================================================
# Step 5 — HMM-HMM Edges: MMseqs2 Profile-Profile Search (fastest mode)
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   sbatch hmm_hmm_edges_mmseqs.slurm
# Tune the USER CONFIGURATION section below before submitting.
#
# Why mmseqs-profile?
# For datasets with thousands to hundreds-of-thousands of subfamilies, the
# MMseqs2 profile-profile search (mmseqs msa2profile + mmseqs search) is
# orders of magnitude faster than pairwise hhalign or db-search hhsearch.
# A dataset that takes 6 days with pairwise hhalign typically finishes in
# 1–4 hours with this mode on a 64-core epyc-64 node.
#
# What it needs:
#   - mmseqs in PATH (already a core pipeline dependency; no extra tools)
#   - MSA (.a3m) files from the build-profiles step (if available, richer
#     profiles are built via mmseqs convertmsa + msa2profile; otherwise the
#     pipeline falls back to iterative sequence search)
#
# Accuracy:
#   - MMseqs2 does not produce HH-suite probability scores; the pipeline uses
#     e-value and coverage thresholds for core/relaxed filtering instead.
#   - For remote homologs with very divergent sequences, pairwise hhalign
#     or db-search can have higher sensitivity; consider running mmseqs-profile
#     as a first pass and re-scoring top hits with hhalign if needed.
#
# Tune in config.yaml:
#   mmseqs.threads: 64       # match --cpus-per-task below
#   hmm_hmm.topN: 200        # max candidate neighbors per subfamily
#   hmm_hmm.max_evalue_relaxed: 1e-3
#   hmm_hmm.mode: mmseqs-profile  (or pass --mode below)
#
# Discovery partition guidance:
#   epyc-64  — 64 cores, 248 GB, ≤48 h  (strongly recommended)
#   largemem — if >500k subfamilies and MMseqs2 needs more than 248 GB
# =============================================================================
#SBATCH --job-name=plm-hmm-mm
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64           # match mmseqs.threads in config.yaml
#SBATCH --mem=128G                   # MMseqs2 uses ~2 GB/thread for profile DB
#SBATCH --time=08:00:00              # most datasets finish in 1–4 h at 64 cores
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these paths before submitting
# ---------------------------------------------------------------------------
RESULTS_ROOT="/path/to/results"
CONFIG="/path/to/config.yaml"

# Use fast node-local scratch for MMseqs2 temporary files
MMSEQS_TMPDIR="${TMPDIR:-/tmp}/mmseqs_mm_$$"
mkdir -p "$MMSEQS_TMPDIR"
# Set mmseqs.tmpdir to this value in config.yaml, or export as env var
# (plm_cluster reads mmseqs.tmpdir from config)
# ---------------------------------------------------------------------------

module load conda
conda activate plm_cluster

plm_cluster hmm-hmm-edges \
    --profile_index   "${RESULTS_ROOT}/02_profiles/subfamily_profile_index.tsv" \
    --candidate_edges "${RESULTS_ROOT}/04_embeddings/embedding_knn_edges.tsv" \
    --outdir          "${RESULTS_ROOT}/03_hmm_hmm_edges" \
    --config          "${CONFIG}" \
    --results_root    "${RESULTS_ROOT}" \
    --mode            mmseqs-profile

# Clean up scratch
rm -rf "$MMSEQS_TMPDIR"
