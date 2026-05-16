#!/bin/bash
# =============================================================================
# Step 1 — MMseqs2 Subfamily Clustering
# USC CARC Discovery cluster example
# =============================================================================
# Usage:
#   sbatch mmseqs_cluster.slurm
# Tune the USER CONFIGURATION section below before submitting.
#
# Discovery partition guidance:
#   main      — up to 48 h, 24–64 cores,  89–248 GB RAM (Xeon / EPYC)
#   epyc-64   — up to 48 h, up to 64 cores, 248 GB RAM  (AMD EPYC zen3)
#   oneweek   — up to 7 days, same node types as main  (for very large inputs)
#   debug     — up to 1 h, useful for testing sbatch submission
#
# This step is CPU + I/O bound.  Using all 64 cores on an epyc-64 node
# finishes most datasets in well under 12 hours.
# =============================================================================
#SBATCH --job-name=plm-mmseqs
#SBATCH --account=YOUR_ACCOUNT       # Replace with your CARC project allocation
#SBATCH --partition=epyc-64
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32           # passed to mmseqs via mmseqs.threads in config
#SBATCH --mem=64G                    # increase to 128G for >500k proteins
#SBATCH --time=12:00:00              # increase to 24:00:00 for very large inputs
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail
mkdir -p logs

# ---------------------------------------------------------------------------
# USER CONFIGURATION — edit these paths before submitting
# ---------------------------------------------------------------------------
PROTEINS_FASTA="/path/to/proteins.faa"
CONFIG="/path/to/config.yaml"
RESULTS_ROOT="/path/to/results"

# Set mmseqs tmpdir to the fast node-local scratch; falls back to /tmp
MMSEQS_TMPDIR="${TMPDIR:-/tmp}/mmseqs_$$"
mkdir -p "$MMSEQS_TMPDIR"
# ---------------------------------------------------------------------------

# Activate conda environment (recommended approach on Discovery)
module load conda
conda activate plm_cluster
# Alternative: if tools are available as modules uncomment below and
# comment out the conda lines above:
#   module load mmseqs2/14.7564

# Propagate allocated core count into the config at runtime so mmseqs uses
# exactly as many threads as Slurm gave us (avoids over-subscription).
# The --config flag points to a YAML where mmseqs.threads is already set;
# adjust that value to match --cpus-per-task above (default 32 here).

plm_cluster mmseqs-cluster \
    --proteins_fasta "${PROTEINS_FASTA}" \
    --outdir         "${RESULTS_ROOT}/01_mmseqs" \
    --config         "${CONFIG}" \
    --results_root   "${RESULTS_ROOT}"

# Clean up scratch
rm -rf "$MMSEQS_TMPDIR"
