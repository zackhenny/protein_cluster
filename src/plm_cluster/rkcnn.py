"""Random K-Conditional Nearest Neighbor (rKCNN) for protein embedding space.

Implements the rKCNN algorithm (PeerJ Computer Science 2025) adapted for
protein clustering.  Uses random feature subspaces of ESM-2 embeddings with
MMseqs2 subfamily labels as supervised classes.

Reference: https://peerj.com/articles/cs-2497/
"""
from __future__ import annotations

import logging
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core rKCNN classifier
# ---------------------------------------------------------------------------

class RKCNN:
    """Random K-Conditional Nearest Neighbor classifier.

    Parameters
    ----------
    n_neighbors : int
        k for the inner kCNN classifier in each subspace.
    n_subspaces : int
        Number of random feature subsets to sample.
    subspace_fraction : float
        Fraction of total features to include in each subspace (0, 1].
    score_threshold : float
        Minimum separation score to retain a subspace (0 keeps all).
    weighting : str
        ``"separation"`` weights subspaces by their separation score;
        ``"uniform"`` gives equal weight to every retained subspace.
    random_state : int or None
        Seed for reproducibility.
    device : str
        ``"cpu"`` or ``"cuda"``/``"cuda:N"`` for GPU acceleration.
    """

    def __init__(
        self,
        n_neighbors: int = 5,
        n_subspaces: int = 50,
        subspace_fraction: float = 0.5,
        score_threshold: float = 0.0,
        weighting: str = "separation",
        random_state: int | None = 42,
        device: str = "cpu",
    ) -> None:
        self.n_neighbors = n_neighbors
        self.n_subspaces = n_subspaces
        self.subspace_fraction = subspace_fraction
        self.score_threshold = score_threshold
        self.weighting = weighting
        self.random_state = random_state
        self.device = device

        # Populated by fit()
        self._subspaces: list[dict[str, Any]] = []
        self._classes: np.ndarray | None = None
        self._is_fitted = False

    # ------------------------------------------------------------------
    # Separation score
    # ------------------------------------------------------------------

    @staticmethod
    def _separation_score_cpu(X_sub: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
        """Between-class / within-class variance ratio (CPU, numpy)."""
        global_mean = X_sub.mean(axis=0)
        between_var = 0.0
        within_var = 0.0
        for c in classes:
            mask = y == c
            X_c = X_sub[mask]
            n_c = X_c.shape[0]
            if n_c == 0:
                continue
            class_mean = X_c.mean(axis=0)
            between_var += n_c * float(((class_mean - global_mean) ** 2).sum())
            within_var += float(((X_c - class_mean) ** 2).sum())
        return between_var / (within_var + 1e-12)

    def _separation_score_gpu(self, X_sub: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
        """Between-class / within-class variance ratio (GPU, PyTorch)."""
        import torch
        dev = self.device
        X_t = torch.from_numpy(X_sub).float().to(dev)
        global_mean = X_t.mean(dim=0)
        between_var = torch.tensor(0.0, device=dev)
        within_var = torch.tensor(0.0, device=dev)
        for c in classes:
            mask = torch.from_numpy(y == c).to(dev)
            X_c = X_t[mask]
            n_c = X_c.shape[0]
            if n_c == 0:
                continue
            class_mean = X_c.mean(dim=0)
            between_var = between_var + n_c * ((class_mean - global_mean) ** 2).sum()
            within_var = within_var + ((X_c - class_mean) ** 2).sum()
        return (between_var / (within_var + 1e-12)).item()

    def _separation_score(self, X_sub: np.ndarray, y: np.ndarray, classes: np.ndarray) -> float:
        if self.device.startswith("cuda"):
            try:
                return self._separation_score_gpu(X_sub, y, classes)
            except Exception:
                pass
        return self._separation_score_cpu(X_sub, y, classes)

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(self, X: np.ndarray, y: np.ndarray) -> "RKCNN":
        """Fit rKCNN on training data.

        Parameters
        ----------
        X : ndarray of shape (n_samples, n_features)
            L2-normalized embeddings.
        y : ndarray of shape (n_samples,)
            Class labels (subfamily IDs encoded as integers).
        """
        rng = np.random.RandomState(self.random_state)
        n_samples, n_features = X.shape
        sub_dim = max(1, int(n_features * self.subspace_fraction))
        self._classes = np.unique(y)

        self._subspaces = []
        for _ in range(self.n_subspaces):
            feat_idx = np.sort(rng.choice(n_features, size=sub_dim, replace=False))
            X_sub = X[:, feat_idx]
            score = self._separation_score(X_sub, y, self._classes)

            if score < self.score_threshold:
                continue

            # Fit a sklearn KNeighborsClassifier on this subspace
            from sklearn.neighbors import KNeighborsClassifier
            k = min(self.n_neighbors, n_samples - 1)
            knn = KNeighborsClassifier(n_neighbors=max(1, k), metric="euclidean")
            knn.fit(X_sub, y)

            self._subspaces.append({
                "feat_idx": feat_idx,
                "score": score,
                "knn": knn,
            })

        if not self._subspaces:
            logger.warning("rKCNN: no subspaces passed the score threshold (%.4f). "
                           "Using all %d subspaces with uniform weighting.",
                           self.score_threshold, self.n_subspaces)
            # Refit without threshold
            rng2 = np.random.RandomState(self.random_state)
            for _ in range(self.n_subspaces):
                feat_idx = np.sort(rng2.choice(n_features, size=sub_dim, replace=False))
                X_sub = X[:, feat_idx]
                score = self._separation_score(X_sub, y, self._classes)
                from sklearn.neighbors import KNeighborsClassifier
                k = min(self.n_neighbors, n_samples - 1)
                knn = KNeighborsClassifier(n_neighbors=max(1, k), metric="euclidean")
                knn.fit(X_sub, y)
                self._subspaces.append({"feat_idx": feat_idx, "score": score, "knn": knn})

        self._is_fitted = True
        logger.debug("rKCNN fitted: %d subspaces retained (of %d sampled), "
                      "%d classes, %d features/subspace",
                      len(self._subspaces), self.n_subspaces,
                      len(self._classes), sub_dim)
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Predict class probabilities via weighted subspace ensemble.

        Parameters
        ----------
        X : ndarray of shape (n_queries, n_features)

        Returns
        -------
        proba : ndarray of shape (n_queries, n_classes)
            Columns correspond to ``self._classes`` (sorted unique labels).
        """
        if not self._is_fitted:
            raise RuntimeError("RKCNN is not fitted. Call fit() first.")

        n_queries = X.shape[0]
        n_classes = len(self._classes)
        agg = np.zeros((n_queries, n_classes), dtype=np.float64)
        total_weight = 0.0

        for sub in self._subspaces:
            w = sub["score"] if self.weighting == "separation" else 1.0
            if w <= 0:
                w = 1e-12  # avoid zero weight
            X_sub = X[:, sub["feat_idx"]]
            proba = sub["knn"].predict_proba(X_sub)  # (n_queries, n_fitted_classes)

            # Map fitted classes to our canonical class order
            fitted_classes = sub["knn"].classes_
            for ci, c in enumerate(fitted_classes):
                idx = np.searchsorted(self._classes, c)
                if idx < n_classes and self._classes[idx] == c:
                    agg[:, idx] += w * proba[:, ci]

            total_weight += w

        if total_weight > 0:
            agg /= total_weight

        return agg

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Return the class with highest aggregated probability."""
        proba = self.predict_proba(X)
        return self._classes[np.argmax(proba, axis=1)]

    @property
    def classes_(self) -> np.ndarray:
        """Sorted unique class labels from training."""
        if self._classes is None:
            raise RuntimeError("RKCNN is not fitted.")
        return self._classes


# ---------------------------------------------------------------------------
# Helper: class centroids
# ---------------------------------------------------------------------------

def compute_class_centroids(X: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute per-class centroids.

    Parameters
    ----------
    X : ndarray (n_samples, n_features)
    labels : ndarray (n_samples,)

    Returns
    -------
    centroids : ndarray (n_classes, n_features)
    unique_labels : ndarray (n_classes,)
    """
    unique_labels = np.unique(labels)
    centroids = np.zeros((len(unique_labels), X.shape[1]), dtype=X.dtype)
    for i, c in enumerate(unique_labels):
        centroids[i] = X[labels == c].mean(axis=0)
    # L2-normalize centroids for inner-product search
    norms = np.linalg.norm(centroids, axis=1, keepdims=True) + 1e-12
    centroids = centroids / norms
    return centroids.astype(np.float32), unique_labels


# ---------------------------------------------------------------------------
# FAISS index builder (shared by knn and rkcnn modes)
# ---------------------------------------------------------------------------

def _build_faiss_index(X_f32: np.ndarray, device: str = "cpu"):
    """Build a FAISS inner-product index, using GPU if requested."""
    import faiss  # type: ignore

    dim = X_f32.shape[1]
    if device.startswith("cuda"):
        gpu_id = 0
        if ":" in device:
            try:
                gpu_id = int(device.split(":")[1])
            except (ValueError, IndexError):
                raise ValueError(
                    f"Invalid CUDA device specification: '{device}'. "
                    "Expected format: 'cuda' or 'cuda:N' where N is an integer."
                )
        try:
            res = faiss.StandardGpuResources()
            index_cpu = faiss.IndexFlatIP(dim)
            index = faiss.index_cpu_to_gpu(res, gpu_id, index_cpu)
        except Exception:
            logger.warning("FAISS-GPU not available; falling back to CPU FAISS index.")
            index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatIP(dim)
    index.add(X_f32)
    return index


# ---------------------------------------------------------------------------
# Pipeline integration
# ---------------------------------------------------------------------------

def rkcnn_candidate_edges(
    X: np.ndarray,
    ids: list[str],
    lens: dict[str, int],
    labels: np.ndarray,
    config: dict,
    logger_: logging.Logger | None = None,
) -> list[dict]:
    """Generate candidate edges using rKCNN.

    Parameters
    ----------
    X : ndarray (N, D)
        L2-normalized embeddings.
    ids : list[str]
        Subfamily IDs corresponding to rows of X.
    lens : dict
        subfamily_id -> representative length in amino acids.
    labels : ndarray (N,)
        Integer-encoded class labels for each embedding.
    config : dict
        Full pipeline config (reads ``config["knn"]``).
    logger_ : Logger, optional

    Returns
    -------
    rows : list[dict]
        Edge rows with schema matching the standard KNN output.
    """
    log = logger_ or logger
    knn_cfg = config["knn"]
    k = max(1, min(int(knn_cfg["k"]), len(ids) - 1))
    min_cosine = float(knn_cfg["min_cosine"])
    min_lr = float(knn_cfg["min_len_ratio"])
    max_lr = float(knn_cfg["max_len_ratio"])
    device = str(knn_cfg.get("device", "cpu"))
    cascade_topn = int(knn_cfg.get("rkcnn_cascade_topn", 500))
    n_subspaces = int(knn_cfg.get("rkcnn_n_subspaces", 50))
    subspace_fraction = float(knn_cfg.get("rkcnn_subspace_fraction", 0.5))
    n_neighbors = int(knn_cfg.get("rkcnn_n_neighbors", 5))
    score_threshold = float(knn_cfg.get("rkcnn_score_threshold", 0.0))
    weighting = str(knn_cfg.get("rkcnn_weighting", "separation"))
    random_state = int(knn_cfg.get("rkcnn_random_state", 42))

    N = X.shape[0]
    log.info("rKCNN candidate generation: N=%d, k=%d, subspaces=%d, cascade_topn=%d",
             N, k, n_subspaces, cascade_topn)

    # Verify L2 normalization
    norms = np.linalg.norm(X, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-4):
        log.warning("Embeddings not L2-normalized; normalizing now.")
        X = X / (norms[:, None] + 1e-12)

    rows: list[dict] = []

    if cascade_topn > 0 and cascade_topn < N:
        # ------------------------------------------------------------------
        # Cascading mode: FAISS pre-filter + per-query rKCNN
        # ------------------------------------------------------------------
        log.info("rKCNN cascading mode: FAISS top-%d pre-filter per query", cascade_topn)

        # Build FAISS index on all embeddings for initial pre-filter
        try:
            import faiss  # type: ignore  # noqa: F811
            index = _build_faiss_index(X.astype(np.float32), device)
            # Retrieve top cascade_topn+1 neighbors (including self)
            search_k = min(cascade_topn + 1, N)
            sims, nbrs = index.search(X.astype(np.float32), search_k)
        except Exception:
            log.warning("FAISS not available for cascade; falling back to sklearn.")
            from sklearn.neighbors import NearestNeighbors
            search_k = min(cascade_topn + 1, N)
            nn = NearestNeighbors(n_neighbors=search_k, metric="cosine")
            nn.fit(X)
            d, nbrs = nn.kneighbors(X)
            sims = 1.0 - d

        # For each query, fit rKCNN on its candidate subset
        for i in range(N):
            q = ids[i]
            # Get candidate indices (exclude self)
            cand_idx = [int(idx) for idx in nbrs[i] if idx != i and 0 <= idx < N]
            if not cand_idx:
                continue

            cand_idx_arr = np.array(cand_idx)
            X_cand = X[cand_idx_arr]
            y_cand = labels[cand_idx_arr]

            # Need at least 2 unique classes and 2 candidates for rKCNN
            unique_classes = np.unique(y_cand)
            if len(unique_classes) < 2 or len(cand_idx) < 2:
                # Fall back to cosine similarity ordering (vectorized)
                cosines = X[i] @ X[cand_idx_arr].T
                for idx_pos, j_idx in enumerate(cand_idx):
                    t = ids[j_idx]
                    if t == q:
                        continue
                    cosine = float(cosines[idx_pos])
                    qlen = int(lens.get(q, 0))
                    tlen = int(lens.get(t, 0))
                    ratio = qlen / max(1, tlen)
                    pass_lr = int(min_lr <= ratio <= max_lr)
                    if cosine >= min_cosine and pass_lr:
                        rows.append({
                            "q_subfamily_id": q, "t_subfamily_id": t,
                            "cosine": cosine, "q_len": qlen, "t_len": tlen,
                            "len_ratio": ratio, "pass_len_ratio": pass_lr,
                        })
                continue

            # Fit rKCNN on candidate subset
            rkcnn = RKCNN(
                n_neighbors=min(n_neighbors, len(cand_idx) - 1),
                n_subspaces=n_subspaces,
                subspace_fraction=subspace_fraction,
                score_threshold=score_threshold,
                weighting=weighting,
                random_state=random_state,
                device=device,
            )
            rkcnn.fit(X_cand, y_cand)

            # Predict on the query point
            proba = rkcnn.predict_proba(X[i:i+1])[0]  # (n_classes,)
            classes = rkcnn.classes_

            # Get top-k classes by probability
            top_k_class_idx = np.argsort(proba)[::-1][:k]

            for ci in top_k_class_idx:
                score = float(proba[ci])
                if score < min_cosine:
                    continue
                target_class = classes[ci]
                # Find the representative(s) in this class from candidate set
                class_mask = y_cand == target_class
                class_indices = cand_idx_arr[class_mask]
                for j_idx in class_indices:
                    t = ids[j_idx]
                    if t == q:
                        continue
                    qlen = int(lens.get(q, 0))
                    tlen = int(lens.get(t, 0))
                    ratio = qlen / max(1, tlen)
                    pass_lr = int(min_lr <= ratio <= max_lr)
                    if pass_lr:
                        rows.append({
                            "q_subfamily_id": q, "t_subfamily_id": t,
                            "cosine": score, "q_len": qlen, "t_len": tlen,
                            "len_ratio": ratio, "pass_len_ratio": pass_lr,
                        })
    else:
        # ------------------------------------------------------------------
        # Global mode: single rKCNN model fitted on all data
        # ------------------------------------------------------------------
        log.info("rKCNN global mode: fitting on all %d samples", N)

        rkcnn = RKCNN(
            n_neighbors=min(n_neighbors, N - 1),
            n_subspaces=n_subspaces,
            subspace_fraction=subspace_fraction,
            score_threshold=score_threshold,
            weighting=weighting,
            random_state=random_state,
            device=device,
        )
        rkcnn.fit(X, labels)
        classes = rkcnn.classes_

        # Predict probabilities for all queries
        proba = rkcnn.predict_proba(X)  # (N, n_classes)

        for i in range(N):
            q = ids[i]
            top_k_class_idx = np.argsort(proba[i])[::-1][:k]

            for ci in top_k_class_idx:
                score = float(proba[i, ci])
                if score < min_cosine:
                    continue
                target_class = classes[ci]
                # Find all samples in this class
                class_mask = labels == target_class
                class_indices = np.where(class_mask)[0]
                for j_idx in class_indices:
                    t = ids[j_idx]
                    if t == q:
                        continue
                    qlen = int(lens.get(q, 0))
                    tlen = int(lens.get(t, 0))
                    ratio = qlen / max(1, tlen)
                    pass_lr = int(min_lr <= ratio <= max_lr)
                    if pass_lr:
                        rows.append({
                            "q_subfamily_id": q, "t_subfamily_id": t,
                            "cosine": score, "q_len": qlen, "t_len": tlen,
                            "len_ratio": ratio, "pass_len_ratio": pass_lr,
                        })

    log.info("rKCNN generated %d candidate edges", len(rows))
    return rows
