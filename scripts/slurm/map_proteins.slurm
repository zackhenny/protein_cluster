#!/bin/bash
# =============================================================================
# Step 8a — Map Proteins to Families
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   sbatch map_proteins.slurm
# Tune the USER CONFIGURATION section below before submitting.
#
# This step aligns all full-length proteins against subfamily representative
# sequences using mmseqs easy-search.  It produces:
#   results/05_domain_hits/protein_family_segments.tsv  — per-protein hits
#   results/05_domain_hits/protein_architecture.tsv     — family architecture strings
#
# Fusion-safe: a greedy overlap filter prevents the same protein region from
# being counted in multiple families.  One protein can legitimately belong to
# multiple families if it has non-overlapping domain segments.
#
# Key config.yaml options (mapping section):
#   profile_cov_min: 0.70    — min fraction of subfamily profile covered
#   min_pident: 0.0          — min percent identity (0 = any)
#   max_evalue: 1e-5         — max E-value for mmseqs hits
#   min_segment_len: 120     — min alignment length (aa)
#   max_overlap_aa: 30       — max allowed overlap between accepted segments
#
# Discovery partition guidance:
#   epyc-64  — recommended; 64 cores, 248 GB, sufficient for most datasets
#   main     — adequate for small datasets (<100k proteins)
# =============================================================================
#SBATCH --job-name=plm-map
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32           # threads passed to mmseqs easy-search
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these paths before submitting
# ---------------------------------------------------------------------------
PROTEINS_FASTA="/path/to/proteins.faa"
RESULTS_ROOT="/path/to/results"
CONFIG="/path/to/config.yaml"

MMSEQS_TMPDIR="${TMPDIR:-/tmp}/mmseqs_map_$$"
mkdir -p "$MMSEQS_TMPDIR"
# ---------------------------------------------------------------------------

module load conda
conda activate plm_cluster

plm_cluster map-proteins-to-families \
    --proteins_fasta               "${PROTEINS_FASTA}" \
    --subfamily_to_family_strict   "${RESULTS_ROOT}/06_family_clustering/subfamily_to_family_strict.tsv" \
    --subfamily_to_family_functional "${RESULTS_ROOT}/06_family_clustering/subfamily_to_family_functional.tsv" \
    --subfamily_map                "${RESULTS_ROOT}/01_mmseqs/subfamily_map.tsv" \
    --outdir                       "${RESULTS_ROOT}/05_domain_hits" \
    --config                       "${CONFIG}" \
    --results_root                 "${RESULTS_ROOT}" \
    --resume

rm -rf "$MMSEQS_TMPDIR"
