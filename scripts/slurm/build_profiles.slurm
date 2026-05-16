#!/bin/bash
# =============================================================================
# Step 2 — Build Subfamily Profiles (MAFFT + HH-suite hhmake)
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   sbatch build_profiles.slurm
# Tune the USER CONFIGURATION section below before submitting.
#
# This step runs MAFFT + hhmake for each subfamily in parallel.
# profiles.parallel_workers in config.yaml controls the concurrency;
# set it equal to --cpus-per-task below.
#
# Discovery partition guidance:
#   epyc-64  — preferred; 64 EPYC cores, 248 GB RAM, 48 h max
#   oneweek  — for very large profile sets (>50k subfamilies) needing >48 h
#   main     — acceptable for small/medium datasets
#
# Memory: each MAFFT process uses ~50–200 MB; with 32 workers peak RSS
# is typically under 16 GB.  64 GB is a conservative upper bound.
# =============================================================================
#SBATCH --job-name=plm-profiles
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32           # match profiles.parallel_workers in config.yaml
#SBATCH --mem=64G
#SBATCH --time=24:00:00              # increase to 48:00:00 for >20k subfamilies
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these paths before submitting
# ---------------------------------------------------------------------------
PROTEINS_FASTA="/path/to/proteins.faa"
SUBFAMILY_MAP="/path/to/results/01_mmseqs/subfamily_map.tsv"
CONFIG="/path/to/config.yaml"
RESULTS_ROOT="/path/to/results"
# ---------------------------------------------------------------------------

module load conda
conda activate plm_cluster
# Alternative: load tools as environment modules and skip conda:
#   module load hhsuite/3.3.0
#   module load mafft/7.505

# --resume skips already-built profiles; safe to resubmit after preemption
plm_cluster build-profiles \
    --proteins_fasta "${PROTEINS_FASTA}" \
    --subfamily_map  "${SUBFAMILY_MAP}" \
    --outdir         "${RESULTS_ROOT}/02_profiles" \
    --config         "${CONFIG}" \
    --results_root   "${RESULTS_ROOT}" \
    --resume
