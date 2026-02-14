# plm-cluster

Annotation-independent, fusion-safe protein clustering pipeline for large-scale datasets.

## What it does
`plm_cluster` separates:
1. **Core domain family identity** (subfamily-level profile similarities -> family clustering), and
2. **Protein architecture** (segment-level family containment in full-length proteins).

This prevents fused/multi-domain proteins from collapsing families incorrectly.

## Repository layout
- `src/plm_cluster/` Python package and CLI
- `envs/` Conda environment
- `container/` Docker and Apptainer/Singularity definitions
- `scripts/slurm/` HPC scripts
- `tests/` unit + smoke tests
- `example_data/` toy proteins
- `docs/config.template.yaml` tunable configuration template

## Required tools and libs
External executables (runtime checked):
- `mmseqs`
- `hhmake`
- `hhsearch` (for HMM-HMM edges)
- `mafft` (default MSA tool)

Python dependencies:
- numpy, pandas, scipy
- scikit-learn (KNN fallback)
- python-igraph + leidenalg
- torch + fair-esm (embedding stage)

Optional:
- FAISS (KNN acceleration)
- `mcl` binary

## Offline ESM requirement
The embedding step **never silently downloads** weights.
Provide `--weights_path` to a preloaded local ESM checkpoint:

```bash
plm_cluster embed \
  --reps_fasta results/01_mmseqs/subfamily_reps.faa \
  --weights_path /path/to/esm2_t33_650M_UR50D.pt \
  --config docs/config.template.yaml
```

## CLI stages
- `mmseqs-cluster`
- `build-profiles`
- `hmm-hmm-edges`
- `embed`
- `knn`
- `merge-graph`
- `cluster-families`
- `map-proteins-to-families`
- `write-matrices`
- `run-all`

## End-to-end example
```bash
plm_cluster run-all \
  --proteins_fasta example_data/toy_proteins.faa \
  --weights_path /path/to/preloaded_esm_weights.pt \
  --config docs/config.template.yaml \
  --results_root results
```

## Output tree
```
results/
  00_inputs/
  01_mmseqs/
  02_profiles/
  03_hmm_hmm_edges/
  04_embeddings/
  05_domain_hits/
  06_family_clustering/
  07_membership_matrices/
  manifests/
  logs/
```

Pipeline writes the required TSV/FASTA/NPY artifacts for:
- subfamily assignments and representatives
- subfamily profiles
- raw + core HMM-HMM edges (coverage-gated)
- embeddings + KNN edges
- merged graph and family clusters (Leiden default)
- protein-to-family segment containment hits
- fusion-aware protein architectures
- sparse matrices for subfamily×protein and family×protein

## Fusion handling policy
- **Family definition** is based on **core identity edges** using bidirectional coverage (`min(qcov, tcov)`), minimizing architecture confounding.
- **Protein membership** is based on **segment containment**; a protein can belong to multiple families.
- **Architecture** is reported as ordered family segments (`famA|famC|...`).

## Default parameter guidance
Good starting points for microbial-scale datasets:
- MMseqs: `min_seq_id=0.5–0.7`, `coverage=0.8`, `cov_mode=1`
- HMM-HMM core edges: `mincov>=0.70`, `prob>=90 OR evalue<=1e-5`, optional `aln_len>=120`
- Embedding KNN: `k=100`, `min_cosine=0.35`, length ratio `[0.5, 2.0]`
- Leiden: `resolution=0.5–1.0`, fixed seed

## SLURM usage
CPU:
- `scripts/slurm/mmseqs_cluster.slurm`
- `scripts/slurm/build_profiles.slurm`
- `scripts/slurm/hmm_hmm_edges.slurm`
- `scripts/slurm/cluster_families.slurm`

Embedding:
- GPU: `scripts/slurm/embed_gpu.slurm`
- CPU: `scripts/slurm/embed_cpu.slurm`

## Containers
- Single container: `container/Dockerfile`
- Split containers:
  - `container/graph_profile.def` (mmseqs/hhsuite/graph stack)
  - `container/embedder.def` (PyTorch + ESM)

## Development
```bash
conda env create -f envs/plm_cluster.yaml
conda activate plm_cluster
pip install -e .
pytest -q
```
