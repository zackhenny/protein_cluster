# plm-cluster

**Annotation-independent, fusion-safe protein clustering pipeline.**

`plm_cluster` groups proteins into evolutionary families without relying on
existing functional annotations.  It separates *core-domain identity* from
*domain architecture* so that fusion proteins do not falsely merge unrelated
families.

```
Input proteins  ──►  MMseqs2 subfamilies  ──►  HMM profiles  ──►  HMM-HMM edges
                                                     │
                               ESM-2 embeddings  ──►  KNN / rKCNN candidates  ──►  Graph merge
                                                                                      │
                                                              Leiden clustering ──►  Family assignments
                                                                                         │
                                                                    Fusion-safe mapping ──►  Matrices & QC
```

## Pipeline overview

| Step | Tool | What it does |
|------|------|-------------|
| 1 | MMseqs2 | Cluster proteins into subfamilies by sequence identity |
| 2 | MAFFT + HH-suite | Build MSAs and profile HMMs per subfamily |
| 3 | ESM-2 (PyTorch) | Generate mean-pooled embeddings for subfamily reps |
| 4 | FAISS / sklearn | Find K-nearest embedding neighbors (KNN or rKCNN mode) |
| 5 | hhalign / hhsearch | Profile-profile alignments on candidate pairs → edge sets |
| 6 | Leiden | Cluster strict and functional family graphs |
| 7 | MMseqs2 | Map full-length proteins to families (fusion-aware segments) |
| 8 | pandas / scipy | Write sparse/dense membership matrices |
| 9 | matplotlib | Generate QC diagnostic plots |

## Quick start

```bash
# 1. Create environment
conda env create -f envs/plm_cluster.yaml
bash scripts/install_in_conda_env.sh plm_cluster . test,embed

# 2. Run the full pipeline
plm_cluster run-all \
  --proteins_fasta proteins.faa \
  --weights_path /path/to/esm2_t33_650M_UR50D.pt \
  --config config.yaml \
  --results_root results
```

> **CPU or GPU?**  Set `embed.device` in your config to `"cpu"` or `"cuda"`.
> CPU is the default and works everywhere; GPU accelerates embedding on large
> datasets.  For the KNN/rKCNN step, set `knn.device: "cuda"` to use
> FAISS-GPU for neighbor search (requires `faiss-gpu`; see
> [Installation](docs/installation_and_containers.md)).

## Key concepts

- **Subfamily** — MMseqs2 cluster of proteins sharing high sequence identity.
- **Strict family (Mode A)** — subfamilies linked by high-confidence, high-coverage
  HMM-HMM alignments.  Use for conservative evolutionary homology groups.
- **Functional family (Mode B)** — broader neighborhoods from relaxed HMM edges
  plus embedding similarity.  Use for exploratory function transfer.
- **Architecture** — the ordered list of family segments per protein.  A protein
  with segments from two different families is a *fusion protein*.

## Configuration

The pipeline uses **YAML** config files exclusively.  A ready-to-edit YAML
config lives in the project root:

```bash
plm_cluster run-all --config config.yaml ...
```

Copy `config.yaml` and edit to taste.  Every
parameter has inline comments explaining its purpose and allowed range.

Key tuning knobs:

| Parameter | Effect |
|-----------|--------|
| `mmseqs.min_seq_id` | Subfamily granularity |
| `hmm_hmm.mode` | `pairwise` (hhalign per pair) or `db-search` (hhsearch against ffindex DB) or `mmseqs-profile` (fast all-vs-all) |
| `knn.mode` | `knn` (cosine KNN) or `rkcnn` (Random K-Conditional Nearest Neighbor) |
| `hmm_hmm.min_prob_core` | Strict family sensitivity |
| `graph.leiden_resolution_*` | Family size (lower → larger families) |
| `embed.device` | CPU or GPU for embeddings |
| `knn.device` | CPU or GPU for KNN/rKCNN neighbor search (requires faiss-gpu) |
| `mapping.min_pident` / `min_segment_len` | Filter noisy/short hits (legacy key `min_prob` still accepted) |
| `profiles.parallel_workers` | Concurrent MAFFT+hhmake jobs (separate from per-tool thread count) |
| `hmm_hmm.parallel_workers` | Concurrent hhalign/hhsearch jobs (separate from per-tool thread count) |

See [`config.yaml`](config.yaml) for the full list.

> **HPC note (CPU oversubscription):** `mmseqs.threads` controls the thread
> count passed to each tool invocation, while `profiles.parallel_workers` and
> `hmm_hmm.parallel_workers` control how many jobs run concurrently.  Set
> `parallel_workers × threads ≤ CPU cores in your allocation` to avoid
> oversubscription.

## CLI commands

```
plm_cluster mmseqs-cluster            # Step 1 (standard)
plm_cluster orthofinder-cluster       # Step 1 (OrthoFinder mode)
plm_cluster build-profiles            # Step 2
plm_cluster embed                     # Step 3
plm_cluster knn                       # Step 4
plm_cluster hmm-hmm-edges             # Step 5
plm_cluster merge-graph               # Step 6
plm_cluster cluster-families          # Step 7
plm_cluster map-proteins-to-families  # Step 8
plm_cluster write-matrices            # Step 9
plm_cluster qc-plots                  # Generate QC figures
plm_cluster run-all                   # Run everything end-to-end (standard)
plm_cluster run-all-orthofinder       # Run everything end-to-end (OrthoFinder mode)
```

To see all options for a subcommand, use `--help`:

```bash
plm_cluster hmm-hmm-edges --help
plm_cluster run-all --help
plm_cluster run-all-orthofinder --help
```

Backward-compatible aliases: `cluster` → `cluster-families`,
`map-proteins` → `map-proteins-to-families`.

## OrthoFinder integration

`plm_cluster` can use [OrthoFinder](https://github.com/OrthoFinder/OrthoFinder)
output as its starting point.  Instead of running global MMseqs2 clustering
(Step 1), proteins are pre-grouped by their HOG or OG membership and then
**subclustered within each group**.  Steps 2–9 run unchanged.

### Why integrate OrthoFinder?

| Benefit | Details |
|---------|---------|
| Orthology-aware subfamilies | Proteins in the same HOG are already orthologous; subclustering separates recent paralogs (lineage-specific expansions) into distinct subfamilies |
| Lower sequence-identity threshold | Within-HOG proteins diverge across species; the default `subcluster_min_seq_id: 0.4` is intentionally lower than the global `mmseqs.min_seq_id: 0.6` |
| Cross-HOG edges preserved | HMM-HMM edges (Step 5) still compare across all HOGs, revealing domain sharing, convergent evolution, and fusion events |
| Biological provenance | `og_subfamily_map.tsv` maps every subfamily back to its source OG for downstream phylogenetic and functional analyses |
| Singleton preservation | Subfamilies with no retained graph edges are preserved as singleton families — no subfamily is dropped |

### Quick start

```bash
# 1. Run OrthoFinder on your proteomes (produces HOG FASTA files)
orthofinder -f proteomes/ -t 16

# 2. Point plm_cluster at the N0 HOG directory (root-level, non-redundant)
plm_cluster run-all-orthofinder \
  --og_dir OrthoFinder/Results_*/Phylogenetic_Hierarchical_Orthogroups/N0/ \
  --weights_path /path/to/esm2_t33_650M_UR50D.pt \
  --config docs/config.orthofinder.yaml \
  --results_root results \
  --resume
```

Or use OG sequences instead of HOGs:

```bash
plm_cluster run-all-orthofinder \
  --og_dir OrthoFinder/Results_*/Orthogroup_Sequences/ \
  --weights_path /path/to/esm2_t33_650M_UR50D.pt \
  --results_root results
```

### HOG level choice

OrthoFinder writes HOGs at every node of the species tree:

| Directory | Level | Recommendation |
|-----------|-------|---------------|
| `N0/` | Species-tree root | **Broadest non-redundant set; equivalent to OGs.  Recommended starting point.** |
| `N1/`, `N2/`, … | Internal nodes | Finer pre-grouping; use when you want to restrict subclustering to a specific clade |
| Leaf directories | Species-specific | Very fine; most subfamilies will be singletons |

### New output files

```
results/01_mmseqs/
  subfamily_map.tsv        # Same format as standard run (protein_id, subfamily_id, is_rep)
  subfamily_reps.faa       # One representative per subfamily
  subfamily_stats.tsv      # Per-subfamily member count, rep ID, rep length
  proteins_combined.faa    # All proteins from all HOGs/OGs (used by downstream steps)
  og_subfamily_map.tsv     # Provenance: (subfamily_id, og_id) — NEW
```

### New config keys (`orthofinder` section)

| Key | Default | Description |
|-----|---------|-------------|
| `subcluster_mode` | `linclust` | `linclust` \| `cluster` \| `auto` — controls which MMseqs2 algorithm runs within each HOG |
| `auto_linclust_min_size` | `1000` | HOG size at or above which `auto` mode switches to `linclust` |
| `subcluster_min_seq_id` | `0.4` | Min sequence identity within an OG for subclustering |
| `subcluster_coverage` | `0.8` | Min alignment coverage for within-OG subclustering |
| `subcluster_cov_mode` | `1` | MMseqs2 coverage mode (0=query, 1=target, 2=bidirectional) |
| `subcluster_alignment_mode` | `3` | MMseqs2 `--alignment-mode` (3 = full alignment; required for cluster-reassign) |
| `subcluster_cluster_reassign` | `false` | Pass `--cluster-reassign` to mmseqs cluster (silently ignored for linclust) |
| `min_og_size_for_subclustering` | `2` | OGs smaller than this skip MMseqs2; each protein becomes a singleton subfamily |
| `subcluster_threads` | `4` | CPU threads per individual MMseqs2 OG invocation |
| `parallel_og_workers` | `4` | OGs processed concurrently (`total CPU ≈ parallel_og_workers × subcluster_threads`) |
| `gene_trees_source` | `""` | Path to OrthoFinder v3 gene-tree directory or file; see [Filtering by gene tree](#filtering-by-gene-tree-orthofinder-v3) |

**Subclustering modes:**

| Mode | When to use |
|------|-------------|
| `linclust` *(default)* | Large datasets; O(n) runtime, minimal memory |
| `cluster` | Small/divergent HOGs; O(n²) but better recall; supports `--cluster-reassign` |
| `auto` | Hybrid: uses `cluster` for HOGs below `auto_linclust_min_size`, `linclust` for larger ones; logs the choice per HOG |

**`--cluster-reassign`** improves cluster membership by reassigning border-zone
members to their nearest cluster center.  Set `subcluster_cluster_reassign: true`
to enable it; it is only applied when the effective algorithm is `cluster`
(silently ignored for `linclust`).

### Filtering by gene tree (OrthoFinder v3)

OrthoFinder v3 resolves gene trees only for a subset of orthogroups.  Use
`--gene-trees-source` (or the `orthofinder.gene_trees_source` config key) to
restrict the pipeline to OrthoGroups that have a resolved gene tree, skipping
all others.

**Directory of `*_tree.txt` files** (OrthoFinder v3 default output):

```bash
plm_cluster run-all-orthofinder \
  --og_dir OrthoFinder/Results_*/Orthogroup_Sequences/ \
  --gene-trees-source OrthoFinder/Results_*/Gene_Trees/Resolved_Gene_Trees/ \
  --weights_path /path/to/esm2_t33_650M_UR50D.pt \
  --results_root results
```

**Combined `Resolved_Gene_Trees.txt` file** (plain list or Newick content):

```bash
plm_cluster run-all-orthofinder \
  --og_dir OrthoFinder/Results_*/Orthogroup_Sequences/ \
  --gene-trees-source OrthoFinder/Results_*/Resolved_Gene_Trees.txt \
  --weights_path /path/to/esm2_t33_650M_UR50D.pt \
  --results_root results
```

Both forms can also be set in the config file under
`orthofinder.gene_trees_source`.  The CLI flag takes precedence.

**Supported file formats for `Resolved_Gene_Trees.txt`:**

| Format | Example line |
|--------|-------------|
| Simple list | `OG0000001` |
| Newick content | `OG0000001: (speciesA_prot1:0.1,speciesB_prot2:0.2);` |

**Provenance chain** — every output contains keys to reconstruct the full join:

```
protein_id  →  subfamily_id       via  01_mmseqs/subfamily_map.tsv
subfamily_id → og_id              via  01_mmseqs/og_subfamily_map.tsv
subfamily_id → strict_family_id   via  06_family_clustering/subfamily_to_family_strict.tsv
subfamily_id → functional_family  via  06_family_clustering/subfamily_to_family_functional.tsv
```

A dedicated OrthoFinder config template is at
[`docs/config.orthofinder.yaml`](docs/config.orthofinder.yaml).

## Resuming interrupted runs and progress logging

Every pipeline stage supports `--resume`.  Add `--resume` to any command to
safely restart it after an interruption; stages that have already completed
their work will skip redundant computation.

| Stage | `--resume` behaviour |
|-------|----------------------|
| `mmseqs-cluster` | Skips the step entirely if its output files already exist |
| `orthofinder-cluster` | Skips the step entirely if output files already exist |
| `build-profiles` | Skips already-built per-subfamily `.hhm` profiles; rebuilds only the missing ones |
| `embed` | Skips entirely if `embeddings.npy` already exists; if `embed.checkpoint_dir` is set, reloads per-batch checkpoints and resumes from the last completed batch *(OOM / timeout recovery)* |
| `knn` | Skips the step entirely if the output KNN TSV already exists |
| `hmm-hmm-edges` | Skips already-completed pairs; reads an NDJSON progress log in real time |
| `merge-hmm-shards` | Skips the step if the merged output already exists |
| `merge-graph` | Skips the step if the merged graph files already exist |
| `cluster-families` | Skips the step if family assignment files already exist |
| `map-proteins-to-families` | Skips the step if protein mapping outputs already exist |
| `write-matrices` | Skips the step if matrix output files already exist |
| `qc-plots` | Skips the step if the QC output directory already contains plots |
| `run-all` / `run-all-orthofinder` | Passes `--resume` to every supported stage above |

### Embedding resumability (OOM / timeout recovery)

Large embedding runs on GPU can be interrupted by out-of-memory errors or
SLURM time limits.  Enable batch-level checkpointing in your config:

```yaml
embed:
  checkpoint_dir: "tmp/embed_checkpoints"   # any writable directory
```

Each completed batch is saved as `embed_batch_NNNNNN.npy`.  Re-run with
`--resume` to reload completed batches and continue from the interruption
point.  Checkpoint files are removed once the final `embeddings.npy` is
written.

```bash
plm_cluster embed \
  --reps_fasta results/01_mmseqs/subfamily_reps.faa \
  --weights_path /path/to/esm2.pt \
  --outdir results/04_embeddings \
  --resume
```

### Resuming the `hmm-hmm-edges` stage (pairwise and db-search modes)

The `hmm-hmm-edges` stage has fine-grained resume support: each processed pair
is appended to an NDJSON progress log in real time, so only incomplete work is
re-run.  This works for both `pairwise` and `db-search` modes — if the
ffindex DB was not built yet, it will be built on resume.

| Scenario | Progress file |
|----------|--------------|
| Single run | `results/03_hmm_hmm_edges/hmm_hmm_progress.ndjson` |
| Sharded run (shard *N*) | `results/03_hmm_hmm_edges/hmm_hmm_progress.shard_N.ndjson` |

```bash
# Resume an interrupted HMM-HMM edge computation (pairwise or db-search)
plm_cluster hmm-hmm-edges \
  --profile_index results/02_profiles/subfamily_profile_index.tsv \
  --candidate_edges results/04_embeddings/embedding_knn_edges.tsv \
  --outdir results/03_hmm_hmm_edges \
  --resume

# Resume a specific shard
plm_cluster hmm-hmm-edges \
  --profile_index results/02_profiles/subfamily_profile_index.tsv \
  --candidate_edges results/04_embeddings/embedding_knn_edges.tsv \
  --outdir results/03_hmm_hmm_edges \
  --shard-id 2 --n-shards 4 --resume
```

Per-command log files are written to `results/logs/` and a full run manifest
(parameters, tool versions, input checksums) is saved to
`results/manifests/run_manifest.json` after every command.

See [`docs/cli_workflow_and_options.md`](docs/cli_workflow_and_options.md) for
detailed resume and sharding guidance.

## Requirements

**External tools** (must be on `$PATH` or set in config):
- `mmseqs` (MMseqs2)
- `hhmake`, `hhalign` (HH-suite — pairwise mode)
- `hhsearch`, `ffindex_build`, `cstranslate` (HH-suite — db-search mode)
- `mafft`

**Python** (≥ 3.10):
- PyTorch + fair-esm (for embeddings)
- numpy, scipy, pandas, scikit-learn
- python-igraph + leidenalg (for Leiden clustering)
- matplotlib (for QC plots, optional)
- FAISS (optional, falls back to sklearn; install `faiss-gpu` for GPU KNN)

## Output layout

```
results/
  01_mmseqs/           # Subfamily clustering (+ og_subfamily_map.tsv in OrthoFinder mode)
  02_profiles/         # MSAs and HMM profiles
  03_hmm_hmm_edges/    # Profile-profile edge tables
  04_embeddings/       # Embedding vectors and KNN edges
  05_domain_hits/      # Protein-to-family mapping
  06_family_clustering/ # Family assignment tables
  07_membership_matrices/ # Sparse/dense matrices
  qc_plots/            # Diagnostic figures
  manifests/           # Run metadata and reproducibility
  logs/                # Per-command log files
```

## Documentation

| Document | Contents |
|----------|----------|
| [Algorithm background](docs/algorithm_background.md) | Scientific rationale and method details |
| [rKCNN implementation plan](docs/rkcnn_implementation_plan.md) | rKCNN integration design and data flow |
| [CLI workflow](docs/cli_workflow_and_options.md) | Step-by-step and `run-all` usage |
| [Config reference](config.yaml) | All parameters with inline comments |
| [OrthoFinder config](docs/config.orthofinder.yaml) | OrthoFinder-specific config template with detailed comments |
| [Output schemas](docs/output_schemas.md) | File formats and column descriptions |
| [Installation](docs/installation_and_containers.md) | Conda, containers, and HPC tips |

## HPC and containers

- SLURM examples: `scripts/slurm/`
- CPU container: `container/container_cpu_graph.def`
- GPU container: `container/container_embedder.def`
- Docker: `container/Dockerfile`

## Testing

```bash
pytest -q
```

Includes unit tests and a mocked-tool smoke test covering the full pipeline.

## License

See repository root.
