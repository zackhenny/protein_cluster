# Algorithm background and design rationale

This pipeline is designed for **annotation-independent** large-scale protein clustering where domain architecture (fusions/multi-domain proteins) can otherwise confound family definitions.

## Why a two-level model?

A single clustering over full-length proteins mixes two biological signals:

1. **Core homology signal** (shared evolutionary domain identity)
2. **Architecture signal** (domain combinations in full-length proteins)

Fusion proteins can bridge unrelated core groups if architecture is used directly in clustering. To prevent this, `plm_cluster` separates:

- **Subfamily**: fine-grained sequence group from MMseqs2
- **Family (strict mode)**: cluster of subfamilies by profile-profile core-domain similarity
- **Family (functional mode)**: broader neighborhoods from relaxed HMM + embeddings
- **Architecture**: ordered list of family segments per protein

## Stage-by-stage method background

## 1) MMseqs2 subfamilies

MMseqs2 provides fast approximate clustering over all proteins. This stage yields:

- `subfamily_map.tsv`
- subfamily representative FASTA (`subfamily_reps.faa`)
- subfamily size/rep stats

Representatives are used downstream to reduce embedding and HMM costs.

## 2) Profile construction (MAFFT + HH-suite)

For each subfamily:

- build MSA (`.a3m`) with MAFFT
- build profile HMM (`.hhm`) with `hhmake`

Why profiles? Profile-profile comparison is much more sensitive than pairwise sequence alone for remote homologs.

## 3) Candidate-gated HMM-HMM edges

Naive all-vs-all profile comparisons are expensive for 10k-40k subfamilies. The pipeline therefore uses candidate pairs (from embedding KNN by default), then runs profile-profile alignments on those pairs only.

Two execution modes are available:

- **Pairwise** (`hmm_hmm.mode: pairwise`, default): Runs one `hhalign` invocation per candidate pair using a thread pool.  Simple and robust.
- **DB-search** (`hmm_hmm.mode: db-search`): Builds an HH-suite ffindex database (both `_hhm` and `_a3m` databases are built for compatibility) from all profiles and runs one `hhsearch` invocation per unique query subfamily.  Faster for dense candidate sets because hhsearch processes all targets in one pass per query.  Requires `hhsearch` and `ffindex_build` to be installed.

Raw edges include alignment statistics (`prob`, `evalue`, `aln_len`, `qcov`, `tcov`).

Two edge sets are derived:

- **Core edges (strict):** high confidence, bidirectional coverage gate
- **Relaxed edges (functional):** lower thresholds for broader neighborhoods

Default strict weight:

```text
edge_weight = (prob / 100) * min(qcov, tcov)
```

## 4) ESM-2 embeddings on representatives

Embeddings are computed on subfamily representatives (not all proteins) for scalability.

- model: ESM-2 from `fair-esm`
- pooling: residue mean pooling
- long sequences: configurable policy (truncate by default)
- offline: explicit local weights path required

These vectors supply high-recall candidate neighbors and optional functional graph edges.

## 4b) rKCNN candidate generation (optional)

When `knn.mode: rkcnn` is set, the pipeline uses **Random K-Conditional Nearest Neighbor (rKCNN)** instead of standard cosine KNN for candidate generation.

**Why rKCNN for protein embeddings?**

ESM-2 embeddings are high-dimensional (1280-D for the 650M parameter model). Standard KNN with cosine similarity treats all dimensions equally, but many dimensions may carry noise that degrades neighbor quality — especially for remote homologs. rKCNN addresses this through:

1. **Random subspace sampling**: Multiple random subsets of embedding dimensions are drawn (e.g., 50 subspaces of 640 dimensions each)
2. **Conditional kNN per subspace**: A k-nearest-neighbor classifier is trained on each subspace using MMseqs2 subfamily IDs as class labels
3. **Separation scoring**: Each subspace is scored by its between-class / within-class variance ratio — subspaces where classes are well-separated get higher weight
4. **Weighted ensemble**: Predictions from high-quality subspaces are aggregated by weighted probability averaging

**Key properties:**
- Operates on full-dimensional ESM-2 embeddings without PCA or dimensionality reduction
- Embeddings are L2-normalized before fitting
- MMseqs2 subfamily IDs (from Step 1) serve as ground-truth class labels
- Output format is identical to standard KNN, so downstream steps are unaffected

**Cascading strategy for scalability:**

With tens of thousands of subfamilies, computing conditional neighbors against all classes is expensive. The "cascading" approach (configured via `knn.rkcnn_cascade_topn`) uses fast FAISS vector search to first retrieve the top-N closest subfamily centroids for each query, then runs rKCNN only over those N candidate classes. This reduces the computational bottleneck from O(all_classes) to O(N) per query.

**Reference:** [PeerJ Computer Science (2025) — Random k Conditional Nearest Neighbor](https://peerj.com/articles/cs-2497/)

## 5) Graph merge and clustering

Two graph modes are produced:

- **Strict graph:** core HMM edges only
- **Functional graph:** relaxed HMM + embeddings merged by policy

Leiden clustering is applied separately to each graph:

- strict families (`famS_*`)
- functional families (`famF_*`)

## 6) Fusion-safe protein mapping

Family definition is done at subfamily/core level, then full-length proteins are mapped with segment-aware outputs:

- per-hit rows with coordinates
- family segments per protein
- architecture strings (`famA|famB|...`)

This allows one protein to belong to multiple families when multiple non-overlapping segments are present.

## Strict vs functional interpretation

- **Strict families**: best for evolutionary core-domain identity and conservative homology groups.
- **Functional families**: useful for neighborhood analyses, broad function transfer hypotheses, and exploratory downstream graph mining.
