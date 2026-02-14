# Output schemas

Core required outputs follow this structure:

- `01_mmseqs/subfamily_map.tsv`, `subfamily_reps.faa`, `subfamily_stats.tsv`
- `02_profiles/subfamily_profile_index.tsv` (+ per-subfamily `.a3m` and `.hhm`)
- `03_hmm_hmm_edges/hmm_hmm_edges_raw.tsv`, `hmm_hmm_edges_core.tsv`, `hmm_hmm_edges_relaxed.tsv`
- `04_embeddings/embeddings.npy`, `ids.txt`, `lengths.tsv`, `metadata.json`, `embedding_knn_edges.tsv`
- `05_domain_hits/protein_vs_profile_hits.tsv`, `protein_family_segments.tsv`, `protein_architectures.tsv`
- `06_family_clustering/subfamily_to_family_strict.tsv`, `subfamily_to_family_functional.tsv`,
  `family_stats_strict.tsv`, `family_stats_functional.tsv`
- `07_membership_matrices/subfamily_x_protein_sparse.tsv`,
  `family_strict_x_protein_sparse.tsv`, `family_functional_x_protein_sparse.tsv`

`06_family_clustering/subfamily_to_family.tsv` is kept as a backward-compatible alias of strict families.
