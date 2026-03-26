# plm-cluster

**Annotation-independent, fusion-safe protein clustering pipeline.**

`plm_cluster` groups proteins into evolutionary families without relying on
existing functional annotations.  It separates *core-domain identity* from
*domain architecture* so that fusion proteins do not falsely merge unrelated
families.

```
Input proteins  ──►  MMseqs2 subfamilies  ──►  HMM profiles  ──►  HMM-HMM edges
                                                     │
                               ESM-2 embeddings  ──►  KNN candidates  ──►  Graph merge
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
| 4 | FAISS / sklearn | Find K-nearest embedding neighbors as candidate pairs |
| 5 | hhalign | Profile-profile alignments on candidate pairs → edge sets |
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
> datasets.

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
| `hmm_hmm.min_prob_core` | Strict family sensitivity |
| `graph.leiden_resolution_*` | Family size (lower → larger families) |
| `embed.device` | CPU or GPU for embeddings |
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

Backward-compatible aliases: `cluster` → `cluster-families`,
`map-proteins` → `map-proteins-to-families`.

## Requirements

**External tools** (must be on `$PATH` or set in config):
- `mmseqs` (MMseqs2)
- `hhmake`, `hhalign` (HH-suite)
- `mafft`

**Python** (≥ 3.10):
- PyTorch + fair-esm (for embeddings)
- numpy, scipy, pandas, scikit-learn
- python-igraph + leidenalg (for Leiden clustering)
- matplotlib (for QC plots, optional)
- FAISS (optional, falls back to sklearn)

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
