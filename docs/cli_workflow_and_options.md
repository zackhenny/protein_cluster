# Workflow, CLI usage, and run options

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

Add `--resume` to pick up where the stage left off.  Progress is recorded in
`results/03_hmm_hmm_edges/hmm_hmm_progress.ndjson` (one JSON record per line).

```bash
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

Individual shards can also use `--resume` to recover from preemption:

```bash
plm_cluster hmm-hmm-edges ... --shard-id 2 --n-shards 4 --resume
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
