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
```

## 2. Build subfamily profiles
```bash
plm_cluster build-profiles \
  --proteins_fasta proteins.faa \
  --subfamily_map results/01_mmseqs/subfamily_map.tsv \
  --outdir results/02_profiles \
  --config my_config.yaml
```

## 3. Embeddings + KNN candidates
```bash
plm_cluster embed \
  --reps_fasta results/01_mmseqs/subfamily_reps.faa \
  --weights_path /path/to/esm2.pt \
  --outdir results/04_embeddings \
  --config my_config.yaml

plm_cluster knn \
  --embeddings results/04_embeddings/embeddings.npy \
  --ids results/04_embeddings/ids.txt \
  --lengths results/04_embeddings/lengths.tsv \
  --out_tsv results/04_embeddings/embedding_knn_edges.tsv \
  --config my_config.yaml
```

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
already have a `"status": "ok"` entry in that log are skipped.

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

### DB-search mode (faster for dense candidate sets)

Use `--mode db-search` (or set `hmm_hmm.mode: db-search` in your config) to
build an HH-suite ffindex database and run one `hhsearch` per query instead of
one `hhalign` per pair.  Requires `hhsearch` and `ffindex_build` in `$PATH`.

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
```

## 6. Cluster families
```bash
plm_cluster cluster-families \
  --merged_edges_strict results/06_family_clustering/merged_edges_strict.tsv \
  --merged_edges_functional results/06_family_clustering/merged_edges_functional.tsv \
  --subfamily_map results/01_mmseqs/subfamily_map.tsv \
  --outdir results/06_family_clustering \
  --config my_config.yaml
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
```

## 8. Membership matrices
```bash
plm_cluster write-matrices \
  --subfamily_map results/01_mmseqs/subfamily_map.tsv \
  --protein_family_segments results/05_domain_hits/protein_family_segments.tsv \
  --outdir results/07_membership_matrices \
  --config my_config.yaml
```

## Major tuning options

- `mmseqs.*`: subfamily granularity and sensitivity
- `hmm_hmm.mode`: `pairwise` (default) or `db-search`
- `hmm_hmm.topN`: candidate density and compute load
- `hmm_hmm.mincov_core/min_prob_core`: strict evolutionary confidence
- `hmm_hmm.mincov_relaxed/min_prob_relaxed`: functional neighborhood breadth
- `graph.edge_weight_policy`: strict/union/gated/downweight_embeddings
- `graph.leiden_resolution_strict` and `graph.leiden_resolution_functional`
- `knn.k` and `knn.min_cosine`
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

| Stage | `--resume` supported | Progress file |
|-------|---------------------|---------------|
| `hmm-hmm-edges` | ✅ Yes | `results/03_hmm_hmm_edges/hmm_hmm_progress[.shard_N].ndjson` |
| `run-all` | ✅ Yes (delegates to `hmm-hmm-edges`) | same as above |
| All other stages | — | (runs are fast or atomic; rerunning is safe) |
