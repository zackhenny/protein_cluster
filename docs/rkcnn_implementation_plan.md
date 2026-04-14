# Implementation Plan: rKCNN Candidate Generation for plm_cluster

## Issue Title
**Add rKCNN (Random K-Conditional Nearest Neighbor) as an alternative/augmentation to KNN candidate generation**

## Summary

Replace or augment the existing KNN candidate generation step (Step 4 / `plm_cluster knn`) with **rKCNN (Random K-Conditional Nearest Neighbor)** — a supervised ensemble method that leverages random feature subspaces to overcome the curse of dimensionality in high-dimensional ESM-2 embedding space.

### Algorithm References

- **Paper**: [PeerJ Computer Science (2025) — Random k Conditional Nearest Neighbor](https://peerj.com/articles/cs-2497/)
- **scikit-learn issue**: [scikit-learn#33265](https://github.com/scikit-learn/scikit-learn/issues/33265#issue-3929106528)
- **Kaggle benchmark**: [rKCNN Benchmarking on 20newsgroup](https://www.kaggle.com/code/djlaurenrittersport/rkcnn-benchmarking-20newsgroup#RKCNN-(Random-K-Conditional-Nearest-Neighbor))

---

## Motivation

The current KNN step uses unsupervised cosine similarity in the full ESM-2 embedding space (1280-dimensional for `esm2_t33_650M_UR50D`). While effective, it suffers from the curse of dimensionality — many embedding dimensions carry noise that degrades neighbor quality for remote homologs.

rKCNN addresses this by:
1. **Random subspace sampling** — training multiple kCNN models on random feature subsets
2. **Separation scoring** — computing between-class / within-class variance ratios to weight discriminative subspaces
3. **Supervised conditioning** — leveraging MMseqs2 subfamily labels as ground-truth classes for "conditional" nearest neighbor computation
4. **Weighted ensemble aggregation** — combining predictions across subspaces, weighting by discriminative power

This should improve candidate recall for remote homologs without requiring dimensionality reduction (PCA/UMAP), preserving the full richness of ESM-2 embeddings.

---

## Design Constraints

1. **No Dimensionality Reduction**: rKCNN operates on full-dimensional data by using the random subspace method. Do NOT apply PCA, UMAP, or any dimensionality reduction to the ESM-2 embeddings.

2. **L2 Normalization**: ESM-2 embeddings must be strictly L2-normalized before passing into rKCNN. The current `knn()` function already does this (line 1078 of `pipeline.py`), but the new rKCNN path must ensure it as well.

3. **Labels = MMseqs2 Subfamilies**: rKCNN requires class labels. Use the MMseqs2 subfamily IDs (from Step 1, `subfamily_map.tsv`) as the ground-truth class labels for each embedding vector. Each subfamily representative's embedding gets its own subfamily ID as the label.

4. **Cascading FAISS Pre-filter (Optional)**: With tens of thousands of MMseqs2 clusters, computing conditional neighbors against all classes is prohibitively expensive. A "cascading" approach should:
   - Use standard FAISS inner-product search to retrieve the top-*N* closest subfamily centroids (or representatives) for each query
   - Execute the rKCNN logic only across those *N* candidate classes
   - This is configurable via `knn.rkcnn_cascade_topn` in the config

---

## Data Flow Overview

```
                       ┌──────────────────────────────────┐
                       │  Step 1: MMseqs2 Clustering       │
                       │  → subfamily_map.tsv              │
                       │  → subfamily_reps.faa             │
                       └──────────────┬───────────────────┘
                                      │
                       ┌──────────────▼───────────────────┐
                       │  Step 3: ESM-2 Embeddings         │
                       │  → embeddings.npy (N × 1280)      │
                       │  → ids.txt                        │
                       └──────────────┬───────────────────┘
                                      │
                  ┌───────────────────▼───────────────────────┐
                  │         Step 4: Candidate Generation       │
                  │                                            │
                  │  mode="knn" (existing, default):           │
                  │    FAISS/sklearn cosine KNN → edges        │
                  │                                            │
                  │  mode="rkcnn" (new):                       │
                  │    1. L2-normalize embeddings               │
                  │    2. Load subfamily labels from            │
                  │       subfamily_map.tsv                     │
                  │    3. (Optional cascade) FAISS top-N        │
                  │       subfamily pre-filter per query        │
                  │    4. rKCNN fit on (embeddings, labels)     │
                  │       using random subspaces                │
                  │    5. For each query, predict_proba over    │
                  │       candidate classes → rank by prob      │
                  │    6. Emit top-k edges with rKCNN scores    │
                  │                                            │
                  │  Output: embedding_knn_edges.tsv (same     │
                  │  schema as current KNN output)              │
                  └───────────────────┬───────────────────────┘
                                      │
                       ┌──────────────▼───────────────────┐
                       │  Step 5: HMM-HMM Edges            │
                       │  (unchanged — uses candidate       │
                       │   edges from Step 4)               │
                       └──────────────────────────────────┘
```

---

## Detailed Implementation Plan

### Phase 1: New Module — `src/plm_cluster/rkcnn.py`

Create a new file `src/plm_cluster/rkcnn.py` containing the rKCNN implementation adapted for the protein clustering use case.

**Contents:**

1. **`class RKCNN`** — Core rKCNN classifier
   - Constructor parameters: `n_neighbors` (k for kCNN), `n_subspaces` (number of random feature subsets, default 50), `subspace_fraction` (fraction of features per subspace, default 0.5), `score_threshold` (minimum separation score to retain a subspace, default 0.0), `weighting` (`"separation"` or `"uniform"`), `random_state` (int, for reproducibility)
   - `fit(X, y)` — For each subspace: sample random feature indices, project training data, compute separation score (between-class variance / within-class variance ratio), fit a `sklearn.neighbors.KNeighborsClassifier` on the projected data
   - `predict_proba(X)` — For each subspace with score ≥ threshold: project query, get class probabilities from the fitted kNN, aggregate via weighted average (weight = separation score) across subspaces
   - `predict(X)` — Returns the class with highest aggregated probability
   - Uses only numpy, scikit-learn (`KNeighborsClassifier`), and the standard library

2. **`def rkcnn_candidate_edges(...)`** — Pipeline integration function
   - Signature: `rkcnn_candidate_edges(X, ids, lens, labels, config, logger) -> list[dict]`
   - Handles L2 normalization verification
   - If cascading is enabled (`rkcnn_cascade_topn > 0`):
     - Build a FAISS inner-product index on the embeddings
     - For each query, retrieve the top-N closest classes (by centroid or representative proximity)
     - Subset the training data to only those N classes
     - Run rKCNN `.predict_proba()` on the subset → extract top-k class predictions with scores
   - If cascading is disabled:
     - Fit one global RKCNN model on all data
     - For each query, use leave-one-out or query-exclusion to get class probabilities
   - Emit edges in the same schema as the current KNN function: `q_subfamily_id`, `t_subfamily_id`, `cosine` (repurposed as rKCNN score), `q_len`, `t_len`, `len_ratio`, `pass_len_ratio`

3. **`def compute_class_centroids(X, labels)`** — Helper to compute per-class centroids for the cascading FAISS pre-filter step

### Phase 2: Modify `src/plm_cluster/pipeline.py`

**Changes to the `knn()` function (lines 1065–1114):**

1. Add a mode selector at the top of `knn()` that reads `config["knn"]["mode"]`:
   - `"knn"` (default) — execute the existing FAISS/sklearn cosine KNN logic (unchanged)
   - `"rkcnn"` — delegate to the new `rkcnn_candidate_edges()` function

2. When mode is `"rkcnn"`:
   - Load `subfamily_map.tsv` to build an `id → subfamily_id` mapping (the labels). Since the embeddings are already computed on subfamily representatives, and `ids.txt` contains the subfamily IDs, the label for each embedding vector IS its own subfamily ID. However, for rKCNN to work meaningfully, we need a *higher-level* grouping. **Clarification**: The "labels" for rKCNN are the subfamily IDs themselves — each embedding point is a subfamily representative, and we are trying to find which *other* subfamilies are most similar. rKCNN treats each subfamily as its own class and uses random subspace conditional proximity to determine nearest class neighbors.

3. Add a new CLI argument `--subfamily_map` to the `knn` subparser (only required when `mode=rkcnn`).

4. The output TSV schema remains identical so downstream steps (`hmm-hmm-edges`, `merge-graph`) require zero changes.

**Changes to the `embed()` function (lines 978–1063):**

- No changes needed. The `embed()` function already produces raw ESM-2 embeddings without dimensionality reduction. L2 normalization is applied in the `knn()` step, which is the correct place.

### Phase 3: Modify `src/plm_cluster/config.py`

**Add rKCNN parameters to `DEFAULT_CONFIG["knn"]`:**

```python
"knn": {
    "mode": "knn",                    # "knn" (default) | "rkcnn"
    "k": 100,
    "min_cosine": 0.35,
    "min_len_ratio": 0.5,
    "max_len_ratio": 2.0,
    # --- rKCNN-specific parameters (only used when mode="rkcnn") ---
    "rkcnn_n_subspaces": 50,          # Number of random feature subspaces
    "rkcnn_subspace_fraction": 0.5,   # Fraction of features per subspace
    "rkcnn_n_neighbors": 5,           # k for the inner kCNN classifier
    "rkcnn_score_threshold": 0.0,     # Min separation score to retain a subspace
    "rkcnn_weighting": "separation",  # "separation" | "uniform"
    "rkcnn_cascade_topn": 500,        # Top-N classes to pre-filter via FAISS (0=disabled)
    "rkcnn_random_state": 42,         # Random seed for subspace sampling
}
```

**Add validation rules to `_RANGE_CHECKS`:**

```python
("knn", "rkcnn_n_subspaces", 1, 10_000),
("knn", "rkcnn_subspace_fraction", 0.01, 1.0),
("knn", "rkcnn_n_neighbors", 1, 10_000),
("knn", "rkcnn_score_threshold", 0.0, 1e6),
("knn", "rkcnn_cascade_topn", 0, 1_000_000),
```

**Add mode validation:**

```python
_VALID_KNN_MODES = {"knn", "rkcnn"}
```

And add a check in `validate_config()` similar to the HMM mode check.

### Phase 4: Modify `src/plm_cluster/cli.py`

**Update the `knn` subparser (around line 61):**

```python
p = sub.add_parser("knn")
add_common(p)
p.add_argument("--embeddings", required=True)
p.add_argument("--ids", required=True)
p.add_argument("--lengths", required=True)
p.add_argument("--out_tsv", default="results/04_embeddings/embedding_knn_edges.tsv")
p.add_argument("--subfamily_map", default=None,
               help="Path to subfamily_map.tsv (required when knn.mode=rkcnn)")
p.add_argument("--mode", default=None, choices=["knn", "rkcnn"],
               help="Candidate generation mode: 'knn' (default) or 'rkcnn'")
p.add_argument("--resume", action="store_true",
               help="Skip this step if the output TSV already exists")
```

**Update the `knn` dispatch (around line 189):**

Pass `subfamily_map` to the `knn()` function so it can load labels when in rkcnn mode.

**Update `run-all` (around line 272):**

Pass `subfamily_map` path (`root / "01_mmseqs/subfamily_map.tsv"`) to the KNN step.

### Phase 5: Update Configuration Files

**`config.yaml` (project root):**

Add rKCNN parameters to the `knn:` section with inline comments:

```yaml
knn:
  # Step 5 — Candidate neighbor generation from embeddings
  # mode: "knn"    — Standard FAISS/sklearn cosine KNN (default, fast)
  #        "rkcnn" — Random K-Conditional Nearest Neighbor (supervised ensemble;
  #                  uses MMseqs2 subfamily labels + random subspaces to find
  #                  discriminative neighbors in high-dimensional embedding space)
  mode: knn
  k: 100
  min_cosine: 0.35
  min_len_ratio: 0.5
  max_len_ratio: 2.0
  # --- rKCNN-specific (only when mode: rkcnn) ---
  rkcnn_n_subspaces: 50        # Number of random feature subspaces to sample
  rkcnn_subspace_fraction: 0.5 # Fraction of embedding dimensions per subspace
  rkcnn_n_neighbors: 5         # k for the inner kCNN classifiers
  rkcnn_score_threshold: 0.0   # Min separation score to keep a subspace (0=keep all)
  rkcnn_weighting: separation  # Subspace weighting: "separation" | "uniform"
  rkcnn_cascade_topn: 500      # FAISS pre-filter: top-N candidate classes per query
                                # (0 = disabled, use all classes — slow for large datasets)
  rkcnn_random_state: 42       # Random seed for subspace sampling
```

**`docs/config.template.yaml`:**

Mirror the same additions with detailed inline documentation.

### Phase 6: Update Documentation

**`README.md`:**
- Update the pipeline overview ASCII diagram to show the KNN/rKCNN branch
- Update the Step 4 row in the pipeline table
- Add `knn.mode` to the "Key tuning knobs" table
- Update CLI commands section if the `knn` command gains new arguments

**`docs/algorithm_background.md`:**
- Add a new section "4b) rKCNN candidate generation" explaining the rKCNN algorithm, why it's beneficial for protein embeddings, and the cascading strategy

**`docs/cli_workflow_and_options.md`:**
- Add rKCNN usage examples to the KNN section
- Document the `--subfamily_map` and `--mode` arguments for `plm_cluster knn`

**`docs/config.template.yaml`:**
- Already covered in Phase 5

### Phase 7: Tests

**`tests/test_rkcnn.py`** (new file):

1. **Unit test `RKCNN` class**: Generate synthetic data with known class structure, fit rKCNN, verify `predict_proba` returns valid probabilities, verify `predict` returns expected classes
2. **Unit test `rkcnn_candidate_edges`**: Mock embeddings + labels, verify output schema matches current KNN output
3. **Test cascading mode**: Verify that with `rkcnn_cascade_topn > 0`, only a subset of classes are considered per query
4. **Test L2 normalization**: Verify embeddings are properly normalized before rKCNN fitting
5. **Integration test**: Add an rKCNN path to the existing smoke test (`test_smoke.py`) — run the pipeline with `knn.mode: rkcnn`

**`tests/test_pipeline.py`:**

- Add a test case for `knn()` with `mode="rkcnn"` in the config

---

## Files Modified / Created

### New files:
| File | Purpose |
|------|---------|
| `src/plm_cluster/rkcnn.py` | rKCNN classifier and pipeline integration function |
| `tests/test_rkcnn.py` | Unit and integration tests for rKCNN |
| `docs/rkcnn_implementation_plan.md` | This plan document |

### Modified files:
| File | Changes |
|------|---------|
| `src/plm_cluster/pipeline.py` | Add mode dispatch in `knn()`, import `rkcnn_candidate_edges` |
| `src/plm_cluster/config.py` | Add rKCNN defaults, validation ranges, mode enum |
| `src/plm_cluster/cli.py` | Add `--subfamily_map`, `--mode` to knn parser; pass to `knn()` and `run-all` |
| `config.yaml` | Add rKCNN parameters to `knn:` section |
| `docs/config.template.yaml` | Add rKCNN parameters with full documentation |
| `docs/algorithm_background.md` | Add rKCNN algorithm section |
| `docs/cli_workflow_and_options.md` | Add rKCNN usage examples |
| `README.md` | Update pipeline diagram, table, tuning knobs |

---

## Recommended Implementation Order

1. **`src/plm_cluster/rkcnn.py`** — Core algorithm (can be developed and tested independently)
2. **`src/plm_cluster/config.py`** — Add defaults and validation
3. **`config.yaml` + `docs/config.template.yaml`** — Config file updates
4. **`src/plm_cluster/pipeline.py`** — Integration into the `knn()` function
5. **`src/plm_cluster/cli.py`** — CLI argument updates
6. **`tests/test_rkcnn.py`** — Tests
7. **Documentation** — README, algorithm_background, cli_workflow_and_options

---

## Open Questions / Considerations

1. **Label semantics**: Since each embedding point IS a subfamily representative, and rKCNN needs classes with multiple members, the natural approach is to use the embedding of each subfamily as a 1-member class. The "conditional nearest neighbor" then computes proximity conditioned on class structure. An alternative is to use a coarser pre-clustering (e.g., a preliminary loose KNN-based grouping) to create multi-member classes. The initial implementation should use subfamily IDs directly and evaluate quality.

2. **Computational cost**: rKCNN with 50 subspaces × N queries × M candidate classes involves significant computation. The cascading FAISS pre-filter (default `cascade_topn=500`) is essential for datasets with >10K subfamilies. Benchmarking should be done on real-scale data.

3. **Score interpretation**: The current pipeline uses cosine similarity as the edge weight from KNN. rKCNN outputs class probabilities (0–1). These should map naturally to the same downstream merge-graph logic since both are [0,1] scores. The output column is still named `cosine` for backward compatibility, but a metadata flag or additional column could be added to distinguish the methods.

4. **Parallel execution**: The rKCNN `fit()` step can be parallelized across subspaces using `joblib` or `concurrent.futures`. The cascading approach is embarrassingly parallel per query. Consider adding `n_jobs` as a config parameter.

5. **Memory**: Storing 50 fitted kNN models (one per subspace) in memory could be significant for large datasets. Consider implementing a streaming/batch prediction mode for the cascading approach.
