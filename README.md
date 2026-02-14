# plm-cluster

Fusion-safe, annotation-independent protein clustering pipeline using MMseqs2 + HH-suite + ESM-2 + Leiden.

## Key design updates
- **No Pfam/InterPro required**.
- **Mode A (strict core-domain families):** cluster from `hmm_hmm_edges_core.tsv` only.
- **Mode B (functional neighborhoods):** cluster from relaxed HMM edges + embedding edges.
- **Fusion-safe mapping:** proteins are assigned by segments and can belong to multiple families.

## CLI subcommands
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

## Required executables
Runtime checks enforce:
- `mmseqs`
- `hhmake`
- `hhalign` (or configure path)
Optional:
- `mcl` (if method selected)

Tool versions are logged and included in `results/manifests/run_manifest.json`.

## Offline embeddings
Embedding never downloads weights silently.
You **must** provide:
- `--weights_path /path/to/preloaded_esm2_weights.pt`

## 200k-protein scaling guidance
- Cluster all proteins with MMseqs2.
- Build embeddings only for **subfamily representatives**.
- Use `knn` output as candidate pairs for `hmm-hmm-edges` (avoid naive all-vs-all):
  - `hmm_hmm.topN: 100-300` depending on budget.
- Keep strict/functional clustering separate.

## End-to-end run
```bash
plm_cluster run-all \
  --proteins_fasta example_data/toy_proteins.faa \
  --weights_path /path/to/esm2_t33_650M_UR50D.pt \
  --config docs/config.template.yaml \
  --results_root results
```

## Outputs
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

Important files:
- `06_family_clustering/subfamily_to_family_strict.tsv`
- `06_family_clustering/subfamily_to_family_functional.tsv`
- `07_membership_matrices/family_strict_x_protein_sparse.tsv`
- `07_membership_matrices/family_functional_x_protein_sparse.tsv`

Migration note:
- `subfamily_to_family.tsv` is retained as alias to strict families for compatibility.

## SLURM
Use scripts in `scripts/slurm/` for CPU (MMseqs2/HH-suite/graph) and GPU/CPU embedding.

## Environments and containers
- Conda: `envs/plm_cluster.yaml`
- Apptainer CPU graph/profile: `container/graph_profile.def`
- Apptainer embedder GPU: `container/embedder.def`
- Single-container alternative: `container/Dockerfile`

## Tests
```bash
pytest -q
```
Includes unit tests and a mocked-tool smoke pipeline test that verifies required stage outputs.
