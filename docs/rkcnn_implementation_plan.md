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
   - **GPU support**: When `knn.device` starts with `"cuda"`, uses PyTorch `torch.cdist` for batch pairwise distances instead of sklearn, and PyTorch variance ops for separation scoring (see Phase 8)

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
    "device": "cpu",                  # "cpu" | "cuda" | "cuda:0" — GPU for FAISS & rKCNN
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
  device: cpu                    # "cpu" | "cuda" — GPU accelerates FAISS index & rKCNN distances
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
| `src/plm_cluster/pipeline.py` | Add mode dispatch in `knn()`, import `rkcnn_candidate_edges`, add `_build_faiss_index()` GPU helper |
| `src/plm_cluster/config.py` | Add rKCNN defaults, `knn.device`, validation ranges, mode enum |
| `src/plm_cluster/cli.py` | Add `--subfamily_map`, `--mode` to knn parser; pass to `knn()` and `run-all` |
| `config.yaml` | Add rKCNN parameters and `knn.device` to `knn:` section |
| `docs/config.template.yaml` | Add rKCNN parameters and `knn.device` with full documentation |
| `docs/algorithm_background.md` | Add rKCNN algorithm section and GPU acceleration table |
| `docs/cli_workflow_and_options.md` | Add rKCNN usage examples and GPU config examples |
| `docs/installation_and_containers.md` | Add GPU FAISS installation guidance |
| `README.md` | Update pipeline diagram, table, tuning knobs, GPU guidance |
| `pyproject.toml` | Add `faiss-gpu` optional dependency |
| `plm_cluster.yaml` | Add `faiss-gpu` install note |

---

## Recommended Implementation Order

1. **`src/plm_cluster/rkcnn.py`** — Core algorithm (can be developed and tested independently)
2. **`src/plm_cluster/config.py`** — Add defaults and validation (including `knn.device`)
3. **`config.yaml` + `docs/config.template.yaml`** — Config file updates
4. **`src/plm_cluster/pipeline.py`** — Integration into the `knn()` function (with GPU FAISS path)
5. **`src/plm_cluster/cli.py`** — CLI argument updates
6. **`tests/test_rkcnn.py`** — Tests (including GPU fallback tests)
7. **Documentation** — README, algorithm_background, cli_workflow_and_options
8. **GPU integration** — Wire `knn.device` through `_build_faiss_index()` and rKCNN GPU paths (Phase 8)

---

## Open Questions / Considerations

1. **Label semantics**: Since each embedding point IS a subfamily representative, and rKCNN needs classes with multiple members, the natural approach is to use the embedding of each subfamily as a 1-member class. The "conditional nearest neighbor" then computes proximity conditioned on class structure. An alternative is to use a coarser pre-clustering (e.g., a preliminary loose KNN-based grouping) to create multi-member classes. The initial implementation should use subfamily IDs directly and evaluate quality.

2. **Computational cost**: rKCNN with 50 subspaces × N queries × M candidate classes involves significant computation. The cascading FAISS pre-filter (default `cascade_topn=500`) is essential for datasets with >10K subfamilies. Benchmarking should be done on real-scale data.

3. **Score interpretation**: The current pipeline uses cosine similarity as the edge weight from KNN. rKCNN outputs class probabilities (0–1). These should map naturally to the same downstream merge-graph logic since both are [0,1] scores. The output column is still named `cosine` for backward compatibility, but a metadata flag or additional column could be added to distinguish the methods.

4. **Parallel execution**: The rKCNN `fit()` step can be parallelized across subspaces using `joblib` or `concurrent.futures`. The cascading approach is embarrassingly parallel per query. Consider adding `n_jobs` as a config parameter.

5. **Memory**: Storing 50 fitted kNN models (one per subspace) in memory could be significant for large datasets. Consider implementing a streaming/batch prediction mode for the cascading approach.

---

## Phase 8: GPU Acceleration

GPU support is critical for both the existing KNN step and the new rKCNN mode at scale. The pipeline should leverage GPU hardware when available for the two most compute-intensive steps: **embedding** (already implemented) and **KNN/rKCNN candidate generation** (new).

### 8.1 New Config Parameter: `knn.device`

Add `knn.device` (default `"cpu"`) to `config.yaml`, `docs/config.template.yaml`, and `config.py` `DEFAULT_CONFIG`:

```yaml
knn:
  device: cpu       # "cpu" | "cuda" | "cuda:0" | "cuda:1" etc.
```

This mirrors the existing `embed.device` pattern and controls where the FAISS index and rKCNN distance computations are executed.

### 8.2 Validation in `config.py`

Add `knn.device` to the validation logic in `validate_config()`:

```python
knn_device = str(cfg.get("knn", {}).get("device", "cpu"))
if knn_device not in _VALID_EMBED_DEVICES and not knn_device.startswith("cuda:"):
    errors.append(f"knn.device='{knn_device}' is not valid; use 'cpu' or 'cuda[:N]'")
```

Reuse the existing `_VALID_EMBED_DEVICES` set since the valid values are the same.

### 8.3 GPU Path for Standard KNN Mode (`knn.mode: knn`)

**Current code** (`pipeline.py:1081–1091`):
```python
try:
    import faiss
    index = faiss.IndexFlatIP(X.shape[1])
    index.add(X.astype(np.float32))
    sims, nbrs = index.search(X.astype(np.float32), k + 1)
except Exception:
    from sklearn.neighbors import NearestNeighbors
    ...
```

**GPU-enhanced replacement:**

```python
device = str(config["knn"].get("device", "cpu"))

def _build_faiss_index(X_f32, device):
    """Build a FAISS inner-product index, using GPU if requested and available."""
    import faiss
    dim = X_f32.shape[1]
    if device.startswith("cuda"):
        gpu_id = 0
        if ":" in device:
            gpu_id = int(device.split(":")[1])
        res = faiss.StandardGpuResources()
        index_cpu = faiss.IndexFlatIP(dim)
        index = faiss.index_cpu_to_gpu(res, gpu_id, index_cpu)
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(X_f32)
    return index

try:
    import faiss
    index = _build_faiss_index(X.astype(np.float32), device)
    sims, nbrs = index.search(X.astype(np.float32), k + 1)
except Exception:
    if device.startswith("cuda"):
        logger.warning("FAISS-GPU not available; falling back to CPU.")
    from sklearn.neighbors import NearestNeighbors
    nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
    nn.fit(X)
    d, nbrs = nn.kneighbors(X)
    sims = 1.0 - d
```

This provides:
- Transparent GPU FAISS usage via `faiss.index_cpu_to_gpu()`
- Graceful fallback to CPU FAISS or sklearn if `faiss-gpu` is not installed
- Multi-GPU device selection via `cuda:0`, `cuda:1`, etc.

### 8.4 GPU Path for rKCNN Mode (`knn.mode: rkcnn`)

The rKCNN implementation should support GPU acceleration at three levels:

#### Level 1: Cascading FAISS Pre-filter (GPU FAISS)
The cascading step (top-N class retrieval) uses the same `_build_faiss_index()` helper described above. When `knn.device: cuda`, the centroid index and all top-N queries run on GPU. This is the highest-impact GPU acceleration point — it converts O(N×D) inner products from CPU to GPU.

#### Level 2: rKCNN Pairwise Distance Computation (PyTorch CUDA)
The inner loop of rKCNN computes distances between query embeddings and per-class training points in each subspace. Replace sklearn `KNeighborsClassifier` with PyTorch-based batch distance computation:

```python
import torch

def _gpu_knn_in_subspace(X_train_sub, y_train, X_query_sub, k, device):
    """GPU-accelerated k-nearest-neighbor in a subspace."""
    X_t = torch.from_numpy(X_train_sub).to(device)
    X_q = torch.from_numpy(X_query_sub).to(device)
    # Batch pairwise L2 distances
    dists = torch.cdist(X_q, X_t, p=2)  # (n_query, n_train)
    topk_dists, topk_idx = dists.topk(k, dim=1, largest=False)
    return topk_idx.cpu().numpy(), topk_dists.cpu().numpy()
```

This is particularly beneficial when:
- `rkcnn_n_subspaces` is large (≥50)
- The number of candidate classes per query is large (>100)
- The subspace dimensionality is high (>300)

When `knn.device: cpu`, the implementation falls back to sklearn `KNeighborsClassifier`.

#### Level 3: Separation Score Computation (PyTorch CUDA)
The between-class / within-class variance ratio can be computed efficiently on GPU using PyTorch:

```python
def _gpu_separation_score(X_sub, y, classes, device):
    """GPU-accelerated separation score for a subspace."""
    X_t = torch.from_numpy(X_sub).float().to(device)
    global_mean = X_t.mean(dim=0)
    between_var = 0.0
    within_var = 0.0
    for c in classes:
        mask = (y == c)
        X_c = X_t[mask]
        n_c = X_c.shape[0]
        if n_c == 0:
            continue
        class_mean = X_c.mean(dim=0)
        between_var += n_c * ((class_mean - global_mean) ** 2).sum()
        within_var += ((X_c - class_mean) ** 2).sum()
    return (between_var / (within_var + 1e-12)).item()
```

### 8.5 GPU Fallback Strategy

The implementation must handle missing GPU dependencies gracefully:

```
knn.device="cuda" requested
  ├─ faiss-gpu available? → Use GpuIndexFlatIP for FAISS operations
  │   └─ No? → Fall back to faiss-cpu IndexFlatIP (log warning)
  │       └─ No faiss at all? → Fall back to sklearn NearestNeighbors
  ├─ torch.cuda available? → Use PyTorch CUDA for rKCNN distances
  │   └─ No? → Fall back to sklearn KNeighborsClassifier (log warning)
  └─ Always log the actual device used at step start
```

### 8.6 Dependencies

**`pyproject.toml`:**
```toml
[project.optional-dependencies]
faiss = ["faiss-cpu>=1.7"]
faiss-gpu = ["faiss-gpu>=1.7"]
```

**`plm_cluster.yaml` (conda):**
```yaml
# Default: faiss-cpu
# For GPU: replace with faiss-gpu from conda-forge
```

PyTorch with CUDA is already a dependency for the embedding step, so no additional PyTorch dependency is needed — only `faiss-gpu` is new.

### 8.7 Logging

When the KNN/rKCNN step starts, log the resolved device:

```python
logger.info("KNN device: %s (FAISS-GPU: %s, PyTorch CUDA: %s)",
            device,
            "available" if _has_faiss_gpu else "not available",
            "available" if torch.cuda.is_available() else "not available")
```

### 8.8 Container Updates

The existing GPU container (`container/container_embedder.def`) should include `faiss-gpu` so that GPU-accelerated KNN is available in the same container used for ESM-2 embedding. This avoids needing two separate GPU containers.

### 8.9 Summary: GPU Acceleration Points

| Pipeline Step | Config Key | CPU Backend | GPU Backend | Speedup Factor (est.) |
|--------------|------------|-------------|-------------|----------------------|
| ESM-2 embedding | `embed.device` | PyTorch CPU | PyTorch CUDA | 5–20× |
| KNN index (standard) | `knn.device` | faiss-cpu / sklearn | faiss-gpu `GpuIndexFlatIP` | 10–50× |
| rKCNN cascade pre-filter | `knn.device` | faiss-cpu | faiss-gpu `GpuIndexFlatIP` | 10–50× |
| rKCNN subspace distances | `knn.device` | sklearn `KNeighborsClassifier` | PyTorch `torch.cdist` on CUDA | 5–20× |
| rKCNN separation scoring | `knn.device` | numpy | PyTorch CUDA variance ops | 2–5× |
