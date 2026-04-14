# Workflow, CLI usage, and run options

> **Tip:** Every subcommand has its own `--help` page.  Run
> `plm_cluster <subcommand> --help` (e.g. `plm_cluster hmm-hmm-edges --help`)
> to see all available flags for that step.

## End-to-end execution

```bash
plm_cluster run-all \
  --proteins_fasta example_data/toy_proteins.faa \
  --weights_path /path/to/esm2_weights.pt \
  --config docs/config.template.yaml \
  --results_root results
```

## Stepwise execution

## 1. Subfamily clustering
```bash
plm_cluster mmseqs-cluster \
  --proteins_fasta proteins.faa \
  --outdir results/01_mmseqs \
  --config my_config.yaml

# Resume: skips the step entirely if output files already exist
plm_cluster mmseqs-cluster \
  --proteins_fasta proteins.faa \
  --outdir results/01_mmseqs \
  --config my_config.yaml \
  --resume
```

## 2. Build subfamily profiles
```bash
plm_cluster build-profiles \
  --proteins_fasta proteins.faa \
  --subfamily_map results/01_mmseqs/subfamily_map.tsv \
  --outdir results/02_profiles \
  --config my_config.yaml

# Resume: rebuilds only missing .hhm profiles; skips already-built ones
plm_cluster build-profiles \
  --proteins_fasta proteins.faa \
  --subfamily_map results/01_mmseqs/subfamily_map.tsv \
  --outdir results/02_profiles \
  --config my_config.yaml \
  --resume
```

## 3. Embeddings + KNN candidates
```bash
plm_cluster embed \
  --reps_fasta results/01_mmseqs/subfamily_reps.faa \
  --weights_path /path/to/esm2.pt \
  --outdir results/04_embeddings \
  --config my_config.yaml

# Resume: skips the step if embeddings.npy already exists
plm_cluster embed \
  --reps_fasta results/01_mmseqs/subfamily_reps.faa \
  --weights_path /path/to/esm2.pt \
  --outdir results/04_embeddings \
  --config my_config.yaml \
  --resume

# Standard KNN mode (default)
plm_cluster knn \
  --embeddings results/04_embeddings/embeddings.npy \
  --ids results/04_embeddings/ids.txt \
  --lengths results/04_embeddings/lengths.tsv \
  --out_tsv results/04_embeddings/embedding_knn_edges.tsv \
  --config my_config.yaml

# rKCNN mode — supervised ensemble using MMseqs2 subfamily labels
# Requires --subfamily_map to provide class labels for each embedding
plm_cluster knn \
  --embeddings results/04_embeddings/embeddings.npy \
  --ids results/04_embeddings/ids.txt \
  --lengths results/04_embeddings/lengths.tsv \
  --subfamily_map results/01_mmseqs/subfamily_map.tsv \
  --out_tsv results/04_embeddings/embedding_knn_edges.tsv \
  --mode rkcnn \
  --config my_config.yaml

# Resume: skips the step if the output TSV already exists
plm_cluster knn \
  --embeddings results/04_embeddings/embeddings.npy \
  --ids results/04_embeddings/ids.txt \
  --lengths results/04_embeddings/lengths.tsv \
  --out_tsv results/04_embeddings/embedding_knn_edges.tsv \
  --config my_config.yaml \
  --resume
```

### rKCNN mode details

rKCNN (Random K-Conditional Nearest Neighbor) uses random feature subspaces
and MMseqs2 subfamily labels to find discriminative neighbors in
high-dimensional ESM-2 embedding space.  Key config parameters:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `knn.mode` | `knn` | `"knn"` or `"rkcnn"` |
| `knn.rkcnn_n_subspaces` | 50 | Random feature subspaces to sample |
| `knn.rkcnn_subspace_fraction` | 0.5 | Fraction of dimensions per subspace |
| `knn.rkcnn_n_neighbors` | 5 | k for inner kCNN classifiers |
| `knn.rkcnn_cascade_topn` | 500 | FAISS pre-filter (0 = use all classes) |
| `knn.rkcnn_weighting` | `separation` | `"separation"` or `"uniform"` |

The cascading strategy (`rkcnn_cascade_topn > 0`) is recommended for datasets
with more than 10,000 subfamilies to avoid computational bottlenecks.

### GPU acceleration for KNN / rKCNN

Set `knn.device: cuda` in your config to use FAISS-GPU for the neighbor
search index.  This accelerates both standard KNN and the rKCNN cascading
pre-filter.  Requires the `faiss-gpu` package.

```yaml
# GPU-accelerated config for both embedding and KNN
embed:
  device: cuda          # ESM-2 forward pass on GPU
knn:
  device: cuda          # FAISS-GPU for neighbor search
  mode: rkcnn           # or "knn" for standard KNN
```

The pipeline auto-detects whether `faiss-gpu` is available.  If `knn.device`
is set to `"cuda"` but `faiss-gpu` is not installed, it falls back to CPU
FAISS or sklearn with a warning.

## 4. Candidate-gated HMM-HMM edges
```bash
plm_cluster hmm-hmm-edges \
  --profile_index results/02_profiles/subfamily_profile_index.tsv \
  --candidate_edges results/04_embeddings/embedding_knn_edges.tsv \
  --outdir results/03_hmm_hmm_edges \
  --config my_config.yaml
```

### Resume after interruption

Add `--resume` to pick up where the stage left off.  Every processed pair is
appended to an NDJSON progress log in real time; on resume, all pairs that
already have a `"status": "ok"` entry in that log are skipped.  This works for
both `pairwise` and `db-search` modes.  If the ffindex DB was not built before
the interruption, it will be built automatically on resume.

**Default progress file locations:**

| Run type | Progress file |
|----------|--------------|
| Single run | `results/03_hmm_hmm_edges/hmm_hmm_progress.ndjson` |
| Sharded run (shard *N*) | `results/03_hmm_hmm_edges/hmm_hmm_progress.shard_N.ndjson` |

Each line in the NDJSON file is a JSON record:

```json
{"ts": 1700000000.1, "q": "subfam_000001", "t": "subfam_000042", "status": "ok", "prob": 98.5, "evalue": 1e-10, "aln_len": 120}
{"ts": 1700000001.2, "q": "subfam_000002", "t": "subfam_000003", "status": "failed", "error": "hhalign exited 1"}
```

Only pairs with `"status": "ok"` are considered complete; failed pairs are
re-tried on the next resume.

```bash
# Resume a single interrupted run
plm_cluster hmm-hmm-edges \
  --profile_index results/02_profiles/subfamily_profile_index.tsv \
  --candidate_edges results/04_embeddings/embedding_knn_edges.tsv \
  --outdir results/03_hmm_hmm_edges \
  --config my_config.yaml \
  --resume
```

### MMseqs2 profile-profile search (fastest for large datasets)

Use `--mode mmseqs-profile` (or `hmm_hmm.mode: mmseqs-profile` in your
config) to run an all-vs-all MMseqs2 profile-profile search instead of
HH-suite hhalign or hhsearch.  For datasets with 220K+ families this is
orders of magnitude faster than the other two modes — the job that takes
6 days with pairwise hhalign typically completes in hours.

Requires only `mmseqs` in `$PATH`, which is already a core pipeline
dependency.  No extra HH-suite tools are needed.  When MSA (a3m) files
are available the pipeline builds richer MMseqs2 profiles via
`mmseqs convertmsa` + `mmseqs msa2profile`; otherwise it falls back to
an iterative sequence search (`--num-iterations 2`).

```bash
plm_cluster hmm-hmm-edges \
  --profile_index results/02_profiles/subfamily_profile_index.tsv \
  --candidate_edges results/04_embeddings/embedding_knn_edges.tsv \
  --outdir results/03_hmm_hmm_edges \
  --config my_config.yaml \
  --mode mmseqs-profile
```

Or set in config:

```yaml
hmm_hmm:
  mode: mmseqs-profile
```

### DB-search mode (parallel hhsearch, accurate for moderate datasets)

Use `--mode db-search` (or set `hmm_hmm.mode: db-search` in your config) to
build an HH-suite ffindex database and run one `hhsearch` per query in
parallel instead of one `hhalign` per pair.  Requires `hhsearch`,
`ffindex_build`, and `cstranslate` in `$PATH`.

Results from both the A→B and B→A hhsearch directions are compared for each
canonical family pair; the direction with the higher HH-suite probability
(lower e-value on tie) is kept, correcting the asymmetric-alignment accuracy
issue present in sequential single-direction search.

```bash
plm_cluster hmm-hmm-edges \
  --profile_index results/02_profiles/subfamily_profile_index.tsv \
  --candidate_edges results/04_embeddings/embedding_knn_edges.tsv \
  --outdir results/03_hmm_hmm_edges \
  --config my_config.yaml \
  --mode db-search
```

### Sharded parallel execution

Split the candidate list across *N* independent jobs using `--shard-id` and
`--n-shards`.  Each shard writes its own raw TSV and progress file; call
`merge-hmm-shards` afterwards to produce the combined outputs.

```bash
# Run 4 shards in parallel (e.g. across 4 SLURM array jobs)
for i in 0 1 2 3; do
  plm_cluster hmm-hmm-edges \
    --profile_index results/02_profiles/subfamily_profile_index.tsv \
    --candidate_edges results/04_embeddings/embedding_knn_edges.tsv \
    --outdir results/03_hmm_hmm_edges \
    --config my_config.yaml \
    --shard-id $i --n-shards 4 &
done
wait

# Merge shard results
plm_cluster merge-hmm-shards \
  --outdir results/03_hmm_hmm_edges \
  --config my_config.yaml
```

Individual shards can also use `--resume` to recover from preemption.  Each
shard has its own progress file (`hmm_hmm_progress.shard_N.ndjson`) so shards
can be restarted independently:

```bash
# Restart only shard 2 after preemption
plm_cluster hmm-hmm-edges \
  --profile_index results/02_profiles/subfamily_profile_index.tsv \
  --candidate_edges results/04_embeddings/embedding_knn_edges.tsv \
  --outdir results/03_hmm_hmm_edges \
  --config my_config.yaml \
  --shard-id 2 --n-shards 4 --resume
```

## 5. Merge graphs (strict + functional)
```bash
plm_cluster merge-graph \
  --hmm_core results/03_hmm_hmm_edges/hmm_hmm_edges_core.tsv \
  --hmm_relaxed results/03_hmm_hmm_edges/hmm_hmm_edges_relaxed.tsv \
  --embedding_edges results/04_embeddings/embedding_knn_edges.tsv \
  --outdir results/06_family_clustering \
  --config my_config.yaml

# Resume: skips the step if merged graph files already exist
plm_cluster merge-graph \
  --hmm_core results/03_hmm_hmm_edges/hmm_hmm_edges_core.tsv \
  --hmm_relaxed results/03_hmm_hmm_edges/hmm_hmm_edges_relaxed.tsv \
  --embedding_edges results/04_embeddings/embedding_knn_edges.tsv \
  --outdir results/06_family_clustering \
  --config my_config.yaml \
  --resume
```

## 6. Cluster families
```bash
plm_cluster cluster-families \
  --merged_edges_strict results/06_family_clustering/merged_edges_strict.tsv \
  --merged_edges_functional results/06_family_clustering/merged_edges_functional.tsv \
  --subfamily_map results/01_mmseqs/subfamily_map.tsv \
  --outdir results/06_family_clustering \
  --config my_config.yaml

# Resume: skips the step if family assignment files already exist
plm_cluster cluster-families \
  --merged_edges_strict results/06_family_clustering/merged_edges_strict.tsv \
  --merged_edges_functional results/06_family_clustering/merged_edges_functional.tsv \
  --subfamily_map results/01_mmseqs/subfamily_map.tsv \
  --outdir results/06_family_clustering \
  --config my_config.yaml \
  --resume
```

## 7. Protein mapping and architectures
```bash
plm_cluster map-proteins-to-families \
  --proteins_fasta proteins.faa \
  --subfamily_to_family_strict results/06_family_clustering/subfamily_to_family_strict.tsv \
  --subfamily_to_family_functional results/06_family_clustering/subfamily_to_family_functional.tsv \
  --subfamily_map results/01_mmseqs/subfamily_map.tsv \
  --outdir results/05_domain_hits \
  --config my_config.yaml

# Resume: skips the step if protein mapping outputs already exist
plm_cluster map-proteins-to-families \
  --proteins_fasta proteins.faa \
  --subfamily_to_family_strict results/06_family_clustering/subfamily_to_family_strict.tsv \
  --subfamily_to_family_functional results/06_family_clustering/subfamily_to_family_functional.tsv \
  --subfamily_map results/01_mmseqs/subfamily_map.tsv \
  --outdir results/05_domain_hits \
  --config my_config.yaml \
  --resume
```

## 8. Membership matrices
```bash
plm_cluster write-matrices \
  --subfamily_map results/01_mmseqs/subfamily_map.tsv \
  --protein_family_segments results/05_domain_hits/protein_family_segments.tsv \
  --outdir results/07_membership_matrices \
  --config my_config.yaml

# Resume: skips the step if matrix output files already exist
plm_cluster write-matrices \
  --subfamily_map results/01_mmseqs/subfamily_map.tsv \
  --protein_family_segments results/05_domain_hits/protein_family_segments.tsv \
  --outdir results/07_membership_matrices \
  --config my_config.yaml \
  --resume
```

## Major tuning options

- `mmseqs.*`: subfamily granularity and sensitivity
- `hmm_hmm.mode`: `pairwise` (default) | `db-search` | `mmseqs-profile` (recommended for 220K+ families)
- `hmm_hmm.topN`: candidate density and compute load
- `hmm_hmm.mincov_core/min_prob_core`: strict evolutionary confidence
- `hmm_hmm.mincov_relaxed/min_prob_relaxed`: functional neighborhood breadth
- `graph.edge_weight_policy`: strict/union/gated/downweight_embeddings
- `graph.leiden_resolution_strict` and `graph.leiden_resolution_functional`
- `knn.k` and `knn.min_cosine`
- `knn.mode`: `knn` (default) | `rkcnn` (Random K-Conditional Nearest Neighbor)
- `knn.rkcnn_cascade_topn`: FAISS pre-filter for rKCNN scalability
- `outputs.write_dense_threshold`

## Backward-compatible command aliases

- `plm_cluster map-proteins` -> `map-proteins-to-families`
- `plm_cluster cluster` -> `cluster-families`

## Logs, manifests, and progress files

Every command writes a timestamped log file to `results/logs/` and saves a
run manifest to `results/manifests/run_manifest.json` on completion.  The
manifest contains the CLI parameters, resolved tool paths and versions, a git
hash, and checksums of the input files — useful for reproducibility.

| File | Description |
|------|-------------|
| `results/logs/<cmd>_<timestamp>.log` | stdout/stderr from the command |
| `results/manifests/run_manifest.json` | Full reproducibility record |
| `results/03_hmm_hmm_edges/hmm_hmm_progress.ndjson` | NDJSON progress for single HMM-HMM run |
| `results/03_hmm_hmm_edges/hmm_hmm_progress.shard_N.ndjson` | Per-shard NDJSON progress |

### Resume support by stage

| Stage | `--resume` supported | Behaviour |
|-------|---------------------|-----------|
| `mmseqs-cluster` | ✅ Yes | Skips step entirely if output files already exist |
| `build-profiles` | ✅ Yes | Rebuilds only missing `.hhm` profiles |
| `embed` | ✅ Yes | Skips step if `embeddings.npy` already exists |
| `knn` | ✅ Yes | Skips step if output KNN TSV already exists |
| `hmm-hmm-edges` | ✅ Yes (fine-grained) | Skips completed pairs via NDJSON log; builds ffindex DB if not yet built; mmseqs-profile mode re-runs from scratch if raw TSV absent |
| `merge-hmm-shards` | ✅ Yes | Skips step if merged output already exists |
| `merge-graph` | ✅ Yes | Skips step if merged graph files already exist |
| `cluster-families` | ✅ Yes | Skips step if family assignment files already exist |
| `map-proteins-to-families` | ✅ Yes | Skips step if protein mapping outputs already exist |
| `write-matrices` | ✅ Yes | Skips step if matrix output files already exist |
| `qc-plots` | ✅ Yes | Skips step if output directory already has plots |
| `run-all` | ✅ Yes | Passes `--resume` to every stage above |
