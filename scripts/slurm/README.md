# Slurm Scripts for plm-cluster

Example Slurm submission scripts for running `plm_cluster` on the
[USC CARC Discovery](https://www.carc.usc.edu/user-guides/hpc-systems/discovery/resource-overview-discovery)
HPC cluster.  Each script is a self-contained template; edit the
**USER CONFIGURATION** block near the top before submitting.

> **Tip:** every script includes `--resume`, so any step can be safely
> re-submitted after preemption or a node failure without repeating
> already-completed work.

---

## Quick-start: one node, whole pipeline

For small-to-medium datasets (~5k–20k subfamilies) that fit on a single node:

```bash
# 1. Edit the USER CONFIGURATION section in run_all.slurm
# 2. Submit:
sbatch run_all.slurm
```

For OrthoFinder-derived inputs, use `run_all_orthofinder.slurm` instead.

---

## Discovery partition reference

| Partition  | Max walltime | Cores/node | RAM/node | GPUs/node          | Best for                          |
|------------|-------------|------------|----------|--------------------|-----------------------------------|
| `main`     | 48 h        | 24–64      | 89–248 GB | —                 | Default; most pipeline steps      |
| `epyc-64`  | 48 h        | 64         | 248 GB   | —                  | CPU-intensive steps (profiles, HMM-HMM, mmseqs) |
| `gpu`      | 48 h        | 20–64      | 60–248 GB | 1–3 (A100/A40/L40S/V100/P100) | Embedding (Step 3), GPU KNN (Step 4) |
| `oneweek`  | 7 days      | 20–64      | 60–248 GB | —                 | Long-running CPU jobs (large embed, large HMM-HMM) |
| `largemem` | 48 h        | 64         | 1,498 GB  | —                 | Datasets needing >248 GB RAM      |
| `debug`    | 1 h         | 24–64      | 89–248 GB | some A40          | Testing sbatch scripts            |

---

## Script inventory

### Full-pipeline scripts

| Script | Description |
|--------|-------------|
| `run_all.slurm` | All 9 steps on a single node (standard MMseqs2 input) |
| `run_all_orthofinder.slurm` | All 9 steps using OrthoFinder HOG/OG sequences as input |

### Step-by-step scripts

| Script | Step | Pipeline command | Notes |
|--------|------|-----------------|-------|
| `mmseqs_cluster.slurm` | 1 | `mmseqs-cluster` | MMseqs2 subfamily clustering |
| `build_profiles.slurm` | 2 | `build-profiles` | MAFFT + hhmake per subfamily |
| `embed_cpu.slurm` | 3 | `embed` | ESM-2 on CPU (slow; use GPU for >5k reps) |
| `embed_gpu.slurm` | 3 | `embed` | ESM-2 on GPU — 10–50× faster |
| `knn.slurm` | 4 | `knn` | KNN / rKCNN candidate generation (CPU) |
| `knn_gpu.slurm` | 4 | `knn` | FAISS-GPU KNN / rKCNN |
| `hmm_hmm_edges.slurm` | 5 | `hmm-hmm-edges` | Single-node pairwise or db-search |
| `hmm_hmm_edges_array.slurm` | 5 | `hmm-hmm-edges` | **Slurm job array** — sharded for large datasets |
| `hmm_hmm_edges_mmseqs.slurm` | 5 | `hmm-hmm-edges` | MMseqs2 profile-profile — fastest mode |
| `merge_hmm_shards.slurm` | 5b | `merge-hmm-shards` | Merges shard outputs from the array job |
| `merge_graph.slurm` | 6 | `merge-graph` | Combines HMM + embedding edges |
| `cluster_families.slurm` | 7 | `cluster-families` | Leiden / MCL graph clustering |
| `map_proteins.slurm` | 8a | `map-proteins-to-families` | mmseqs easy-search protein mapping |
| `write_matrices.slurm` | 8b | `write-matrices` | Sparse membership matrix output |
| `orthofinder_cluster.slurm` | 1 (OF) | `orthofinder-cluster` | Within-OG subclustering (standalone) |

---

## Recommended workflows by dataset size

### Small dataset (<5k subfamilies)

A single `run_all.slurm` job on `main` or `epyc-64` is sufficient.
Set `hmm_hmm.mode: pairwise` and `embed.device: cpu` in `config.yaml`.

### Medium dataset (5k–50k subfamilies)

1. Run embed on GPU first (`embed_gpu.slurm`).
2. Submit `run_all.slurm --resume` to skip the completed embed step.
3. Or switch to `hmm_hmm.mode: mmseqs-profile` in `config.yaml` for faster HMM-HMM.

### Large dataset (>50k subfamilies)

Run each step independently; use sharding for HMM-HMM:

```bash
# Step 1: MMseqs2 clustering
sbatch mmseqs_cluster.slurm

# Step 2: Build profiles
sbatch build_profiles.slurm

# Step 3: GPU embedding (much faster for large datasets)
sbatch embed_gpu.slurm

# Step 4: KNN (optionally GPU-accelerated)
sbatch knn.slurm          # or knn_gpu.slurm

# Step 5 option A — MMseqs2 profile-profile (recommended for >50k subfamilies)
sbatch hmm_hmm_edges_mmseqs.slurm

# Step 5 option B — Sharded pairwise (if HH-suite accuracy is required)
ARRAY_JOB_ID=$(sbatch --parsable hmm_hmm_edges_array.slurm)
sbatch --dependency=afterok:${ARRAY_JOB_ID} merge_hmm_shards.slurm

# Steps 6–8b
sbatch merge_graph.slurm
sbatch cluster_families.slurm
sbatch map_proteins.slurm
sbatch write_matrices.slurm
```

---

## HMM-HMM mode selection guide

| Mode | Script | Best for | Requirements |
|------|--------|----------|--------------|
| `pairwise` | `hmm_hmm_edges.slurm` or `hmm_hmm_edges_array.slurm` | Any size; highest HH-suite accuracy | `hhalign` |
| `db-search` | `hmm_hmm_edges.slurm` or `hmm_hmm_edges_array.slurm` | Dense candidate sets; asymmetric-alignment correction | `hhsearch`, `ffindex_build`, `cstranslate` |
| `mmseqs-profile` | `hmm_hmm_edges_mmseqs.slurm` | **Large datasets (>5k subfamilies); 10–100× faster** | `mmseqs` (already required) |

Override the config mode at submission time with `--mode` in the script's `HMM_MODE` variable.

---

## Environment setup

All scripts use conda activation by default:

```bash
module load conda
conda activate plm_cluster
```

Alternatively, load individual tools as environment modules and skip conda:

```bash
module load mmseqs2/14.7564
module load hhsuite/3.3.0
module load mafft/7.505
```

---

## Common `config.yaml` settings for HPC

```yaml
mmseqs:
  threads: 32         # match --cpus-per-task in mmseqs_cluster.slurm
  tmpdir: ""          # set to $TMPDIR in the slurm script (done automatically)

profiles:
  parallel_workers: 32  # match --cpus-per-task in build_profiles.slurm

hmm_hmm:
  mode: mmseqs-profile  # fastest; change to pairwise for HH-suite accuracy
  parallel_workers: 32  # match --cpus-per-task in hmm_hmm_edges.slurm

embed:
  device: cuda          # set when using embed_gpu.slurm
  batch_size: 16        # increase from 4 when GPU has ≥16 GB VRAM

knn:
  device: cuda          # set when using knn_gpu.slurm
```
