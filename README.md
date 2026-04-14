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
  --config docs/config.template.yaml \
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

Copy `config.yaml` (or `docs/config.template.yaml`) and edit to taste.  Every
parameter has inline comments explaining its purpose and allowed range.

Key tuning knobs:

| Parameter | Effect |
|-----------|--------|
| `mmseqs.min_seq_id` | Subfamily granularity |
| `hmm_hmm.mode` | `pairwise` (hhalign per pair) or `db-search` (hhsearch against ffindex DB) |
| `knn.mode` | `knn` (cosine KNN) or `rkcnn` (Random K-Conditional Nearest Neighbor) |
| `hmm_hmm.min_prob_core` | Strict family sensitivity |
| `graph.leiden_resolution_*` | Family size (lower → larger families) |
| `embed.device` | CPU or GPU for embeddings |
| `knn.device` | CPU or GPU for KNN/rKCNN neighbor search (requires faiss-gpu) |
| `mapping.min_prob` / `min_segment_len` | Filter noisy/short hits |

See [`docs/config.template.yaml`](docs/config.template.yaml) for the full list.

## CLI commands

```
plm_cluster mmseqs-cluster       # Step 1
plm_cluster build-profiles       # Step 2
plm_cluster embed                # Step 3
plm_cluster knn                  # Step 4
plm_cluster hmm-hmm-edges        # Step 5
plm_cluster merge-graph          # Step 6
plm_cluster cluster-families     # Step 7
plm_cluster map-proteins-to-families  # Step 8
plm_cluster write-matrices       # Step 9
plm_cluster qc-plots             # Generate QC figures
plm_cluster run-all              # Run everything end-to-end
```

To see all options for a subcommand, use `--help`:

```bash
plm_cluster hmm-hmm-edges --help
plm_cluster run-all --help
```

Backward-compatible aliases: `cluster` → `cluster-families`,
`map-proteins` → `map-proteins-to-families`.

## Resuming interrupted runs and progress logging

Every pipeline stage supports `--resume`.  Add `--resume` to any command to
safely restart it after an interruption; stages that have already completed
their work will skip redundant computation.

| Stage | `--resume` behaviour |
|-------|----------------------|
| `mmseqs-cluster` | Skips the step entirely if its output files already exist |
| `build-profiles` | Skips already-built per-subfamily `.hhm` profiles; rebuilds only the missing ones |
| `embed` | Skips the step entirely if `embeddings.npy` already exists |
| `knn` | Skips the step entirely if the output KNN TSV already exists |
| `hmm-hmm-edges` | Skips already-completed pairs; reads an NDJSON progress log in real time |
| `merge-hmm-shards` | Skips the step if the merged output already exists |
| `merge-graph` | Skips the step if the merged graph files already exist |
| `cluster-families` | Skips the step if family assignment files already exist |
| `map-proteins-to-families` | Skips the step if protein mapping outputs already exist |
| `write-matrices` | Skips the step if matrix output files already exist |
| `qc-plots` | Skips the step if the QC output directory already contains plots |
| `run-all` | Passes `--resume` to every supported stage above |

```bash
# Resume any individual stage — just add --resume
plm_cluster mmseqs-cluster       --proteins_fasta proteins.faa --resume
plm_cluster build-profiles       --proteins_fasta proteins.faa \
                                  --subfamily_map results/01_mmseqs/subfamily_map.tsv --resume
plm_cluster embed                --reps_fasta results/01_mmseqs/subfamily_reps.faa \
                                  --weights_path /path/to/esm2.pt --resume
plm_cluster knn                  --embeddings results/04_embeddings/embeddings.npy \
                                  --ids results/04_embeddings/ids.txt \
                                  --lengths results/04_embeddings/lengths.tsv \
                                  --out_tsv results/04_embeddings/embedding_knn_edges.tsv --resume
plm_cluster hmm-hmm-edges        --profile_index results/02_profiles/subfamily_profile_index.tsv \
                                  --candidate_edges results/04_embeddings/embedding_knn_edges.tsv \
                                  --outdir results/03_hmm_hmm_edges --resume
plm_cluster merge-hmm-shards     --outdir results/03_hmm_hmm_edges --resume
plm_cluster merge-graph          --hmm_core results/03_hmm_hmm_edges/hmm_hmm_edges_core.tsv \
                                  --hmm_relaxed results/03_hmm_hmm_edges/hmm_hmm_edges_relaxed.tsv \
                                  --embedding_edges results/04_embeddings/embedding_knn_edges.tsv \
                                  --outdir results/06_family_clustering --resume
plm_cluster cluster-families     --merged_edges_strict results/06_family_clustering/merged_edges_strict.tsv \
                                  --merged_edges_functional results/06_family_clustering/merged_edges_functional.tsv \
                                  --subfamily_map results/01_mmseqs/subfamily_map.tsv \
                                  --outdir results/06_family_clustering --resume
plm_cluster map-proteins-to-families \
                                  --proteins_fasta proteins.faa \
                                  --subfamily_to_family_strict results/06_family_clustering/subfamily_to_family_strict.tsv \
                                  --subfamily_to_family_functional results/06_family_clustering/subfamily_to_family_functional.tsv \
                                  --subfamily_map results/01_mmseqs/subfamily_map.tsv \
                                  --outdir results/05_domain_hits --resume
plm_cluster write-matrices       --subfamily_map results/01_mmseqs/subfamily_map.tsv \
                                  --protein_family_segments results/05_domain_hits/protein_family_segments.tsv \
                                  --outdir results/07_membership_matrices --resume

# Or resume the entire pipeline end-to-end
plm_cluster run-all \
  --proteins_fasta proteins.faa \
  --weights_path /path/to/esm2.pt \
  --config docs/config.template.yaml \
  --results_root results \
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
- `hhsearch`, `ffindex_build` (HH-suite — db-search mode)
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
  01_mmseqs/           # Subfamily clustering
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
| [Config reference](docs/config.template.yaml) | All parameters with inline comments |
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
