# plm-cluster

`plm_cluster` is a reproducible, annotation-independent, fusion-safe protein clustering pipeline for large-scale datasets.

It combines:
- MMseqs2 subfamily clustering
- HH-suite profile construction and HMM-HMM edges
- ESM-2 embeddings on subfamily representatives
- Graph clustering (Leiden)
- Segment-based protein-family mapping for fusion-aware membership

## Documentation map

- Algorithm background and design: `docs/algorithm_background.md`
- Installation and container builds: `docs/installation_and_containers.md`
- CLI workflow and run options: `docs/cli_workflow_and_options.md`
- Config template: `docs/config.template.yaml`
- Output schemas: `docs/output_schemas.md`

## Core concepts

- **Subfamily**: MMseqs2 cluster of proteins
- **Strict family (Mode A)**: core-domain identity based on stringent HMM-HMM evidence
- **Functional family (Mode B)**: broader neighborhood from relaxed HMM + embedding evidence
- **Architecture**: ordered family segment composition of a full-length protein

This separation prevents fusion proteins from falsely merging core-domain families.

## CLI commands

- `mmseqs-cluster`
- `build-profiles`
- `embed`
- `knn`
- `hmm-hmm-edges`
- `merge-graph`
- `cluster-families`
- `map-proteins-to-families`
- `write-matrices`
- `run-all`

Backward-compatible aliases:
- `cluster` -> `cluster-families`
- `map-proteins` -> `map-proteins-to-families`

## Requirements

External tools checked at runtime:
- `mmseqs`
- `hhmake`
- `hhalign` (or equivalent HH-suite setup)

Optional:
- `mcl` (if selected)

Python stack:
- PyTorch + fair-esm
- numpy/scipy/pandas
- scikit-learn (FAISS optional)
- python-igraph + leidenalg

## Quick start

```bash
conda env create -f envs/plm_cluster.yaml
conda activate plm_cluster
pip install -e .

plm_cluster run-all \
  --proteins_fasta example_data/toy_proteins.faa \
  --weights_path /path/to/esm2_t33_650M_UR50D.pt \
  --config docs/config.template.yaml \
  --results_root results
```

## Output layout

```
results/
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

## Compatibility note

`06_family_clustering/subfamily_to_family.tsv` is preserved as a strict-family alias for older consumers.

## HPC and containers

- SLURM examples: `scripts/slurm/`
- CPU graph/profile Apptainer: `container/container_cpu_graph.def` (alias of `graph_profile.def`)
- GPU embedder Apptainer: `container/container_embedder.def` (alias of `embedder.def`)
- Single container: `container/Dockerfile`

## Testing

```bash
pytest -q
```

Includes unit tests and a mocked-tool smoke workflow test.
