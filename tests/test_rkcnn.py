"""Tests for the rKCNN module and its integration with the KNN pipeline step."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pytest

from plm_cluster.config import DEFAULT_CONFIG, load_config, validate_config
from plm_cluster.rkcnn import RKCNN, _build_faiss_index, compute_class_centroids, rkcnn_candidate_edges


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_synthetic_data(n_classes: int = 5, samples_per_class: int = 10,
                         n_features: int = 64, seed: int = 42):
    """Generate synthetic L2-normalized data with known class structure."""
    rng = np.random.RandomState(seed)
    X_parts = []
    y_parts = []
    for c in range(n_classes):
        center = rng.randn(n_features) * 3
        pts = center + rng.randn(samples_per_class, n_features) * 0.3
        X_parts.append(pts)
        y_parts.append(np.full(samples_per_class, c))
    X = np.vstack(X_parts)
    y = np.concatenate(y_parts)
    # L2-normalize
    norms = np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
    X = X / norms
    return X.astype(np.float32), y


# ---------------------------------------------------------------------------
# RKCNN class unit tests
# ---------------------------------------------------------------------------

class TestRKCNN:
    def test_fit_predict_basic(self):
        X, y = _make_synthetic_data()
        model = RKCNN(n_neighbors=3, n_subspaces=10, subspace_fraction=0.5,
                       random_state=42)
        model.fit(X, y)
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)
        # The model should predict the correct class for most training samples
        accuracy = (preds == y).mean()
        assert accuracy > 0.5, f"Expected >50% accuracy, got {accuracy:.2%}"

    def test_predict_proba_valid(self):
        X, y = _make_synthetic_data()
        model = RKCNN(n_neighbors=3, n_subspaces=10, subspace_fraction=0.5,
                       random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        # Shape: (n_samples, n_classes)
        assert proba.shape == (X.shape[0], len(np.unique(y)))
        # Probabilities should be non-negative
        assert (proba >= 0).all()
        # Each row should sum to approximately 1
        row_sums = proba.sum(axis=1)
        np.testing.assert_allclose(row_sums, 1.0, atol=1e-6)

    def test_predict_proba_not_fitted_raises(self):
        model = RKCNN()
        with pytest.raises(RuntimeError, match="not fitted"):
            model.predict_proba(np.zeros((1, 10)))

    def test_classes_property(self):
        X, y = _make_synthetic_data(n_classes=3)
        model = RKCNN(n_neighbors=2, n_subspaces=5, random_state=0)
        model.fit(X, y)
        np.testing.assert_array_equal(model.classes_, np.array([0, 1, 2]))

    def test_classes_not_fitted_raises(self):
        model = RKCNN()
        with pytest.raises(RuntimeError, match="not fitted"):
            _ = model.classes_

    def test_uniform_weighting(self):
        X, y = _make_synthetic_data()
        model = RKCNN(n_neighbors=3, n_subspaces=10, subspace_fraction=0.5,
                       weighting="uniform", random_state=42)
        model.fit(X, y)
        proba = model.predict_proba(X)
        assert proba.shape[0] == X.shape[0]
        # Should still produce valid probabilities
        assert (proba >= 0).all()

    def test_high_score_threshold_fallback(self):
        """When score_threshold is very high, all subspaces are rejected and
        the model falls back to keeping all subspaces with uniform weight."""
        X, y = _make_synthetic_data()
        model = RKCNN(n_neighbors=3, n_subspaces=5, score_threshold=1e12,
                       random_state=42)
        model.fit(X, y)
        # Should still have subspaces (fallback)
        assert len(model._subspaces) > 0
        preds = model.predict(X)
        assert preds.shape == (X.shape[0],)

    def test_single_class_data(self):
        """Edge case: all samples belong to one class."""
        X = np.random.RandomState(0).randn(20, 16).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        y = np.zeros(20, dtype=int)
        model = RKCNN(n_neighbors=3, n_subspaces=5, random_state=0)
        model.fit(X, y)
        proba = model.predict_proba(X)
        # With only one class, all probabilities should be 1 for that class
        np.testing.assert_allclose(proba[:, 0], 1.0, atol=0.01)


# ---------------------------------------------------------------------------
# compute_class_centroids
# ---------------------------------------------------------------------------

class TestComputeClassCentroids:
    def test_basic(self):
        X = np.array([[1, 0], [0, 1], [1, 1], [0, 0]], dtype=np.float32)
        labels = np.array([0, 0, 1, 1])
        centroids, ulabels = compute_class_centroids(X, labels)
        assert centroids.shape == (2, 2)
        np.testing.assert_array_equal(ulabels, [0, 1])
        # Centroids should be L2-normalized
        norms = np.linalg.norm(centroids, axis=1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)


# ---------------------------------------------------------------------------
# rkcnn_candidate_edges integration
# ---------------------------------------------------------------------------

class TestRkcnnCandidateEdges:
    def _make_pipeline_inputs(self, tmp_path: Path, n: int = 20, d: int = 32):
        """Create synthetic pipeline inputs and return (X, ids, lens, labels, config)."""
        rng = np.random.RandomState(42)
        X = rng.randn(n, d).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        ids = [f"subfam_{i:04d}" for i in range(n)]
        lens = {sid: rng.randint(100, 500) for sid in ids}
        labels = np.arange(n)  # Each sample is its own class
        config = {
            "knn": {
                "mode": "rkcnn",
                "k": 5,
                "min_cosine": 0.0,
                "min_len_ratio": 0.0,
                "max_len_ratio": 100.0,
                "device": "cpu",
                "rkcnn_n_subspaces": 5,
                "rkcnn_subspace_fraction": 0.5,
                "rkcnn_n_neighbors": 3,
                "rkcnn_score_threshold": 0.0,
                "rkcnn_weighting": "separation",
                "rkcnn_cascade_topn": 0,
                "rkcnn_random_state": 42,
            }
        }
        return X, ids, lens, labels, config

    def test_output_schema_global_mode(self, tmp_path):
        X, ids, lens, labels, config = self._make_pipeline_inputs(tmp_path)
        config["knn"]["rkcnn_cascade_topn"] = 0  # global mode
        rows = rkcnn_candidate_edges(X, ids, lens, labels, config)
        assert isinstance(rows, list)
        if rows:
            row = rows[0]
            assert "q_subfamily_id" in row
            assert "t_subfamily_id" in row
            assert "cosine" in row
            assert "q_len" in row
            assert "t_len" in row
            assert "len_ratio" in row
            assert "pass_len_ratio" in row

    def test_output_schema_cascade_mode(self, tmp_path):
        X, ids, lens, labels, config = self._make_pipeline_inputs(tmp_path)
        config["knn"]["rkcnn_cascade_topn"] = 10  # cascade mode
        rows = rkcnn_candidate_edges(X, ids, lens, labels, config)
        assert isinstance(rows, list)
        if rows:
            row = rows[0]
            assert "q_subfamily_id" in row
            assert "cosine" in row

    def test_no_self_edges(self, tmp_path):
        X, ids, lens, labels, config = self._make_pipeline_inputs(tmp_path)
        config["knn"]["rkcnn_cascade_topn"] = 10
        rows = rkcnn_candidate_edges(X, ids, lens, labels, config)
        for row in rows:
            assert row["q_subfamily_id"] != row["t_subfamily_id"]

    def test_cosine_field_is_actual_cosine_not_probability(self, tmp_path):
        """cosine field must contain actual cosine similarity in [-1,1], not rKCNN
        probability scores.  With min_cosine=0.35 and k=5 neighbors, rKCNN
        probabilities (~0.2) would yield zero edges; actual cosines must be used."""
        rng = np.random.RandomState(7)
        n = 30
        X = rng.randn(n, 32).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        ids = [f"s{i}" for i in range(n)]
        lens = {s: 200 for s in ids}
        labels = np.arange(n)
        # Use a realistic min_cosine threshold (default is 0.35)
        config = {
            "knn": {
                "mode": "rkcnn", "k": 5, "min_cosine": 0.35,
                "min_len_ratio": 0.0, "max_len_ratio": 100.0,
                "device": "cpu", "rkcnn_n_subspaces": 5,
                "rkcnn_subspace_fraction": 0.5, "rkcnn_n_neighbors": 5,
                "rkcnn_score_threshold": 0.0, "rkcnn_weighting": "separation",
                "rkcnn_cascade_topn": 15, "rkcnn_random_state": 7,
            }
        }
        rows = rkcnn_candidate_edges(X, ids, lens, labels, config)
        # cosine values should be in [-1, 1] (real cosine similarities)
        for row in rows:
            assert -1.0 <= row["cosine"] <= 1.0, (
                f"cosine={row['cosine']} is outside the valid cosine similarity range "
                "[-1, 1]; the field may be storing rKCNN probability scores instead"
            )
        # With min_cosine=0.35, we should still get some edges (real cosines can
        # exceed 0.35) — not zero edges as probabilities ~0.2 would cause.
        assert len(rows) > 0, (
            "No edges generated; cosine field may be storing probabilities (<0.35) "
            "instead of actual cosine similarities"
        )

    def test_cosine_field_is_actual_cosine_global_mode(self, tmp_path):
        """Same cosine-vs-probability check for global mode (cascade_topn=0)."""
        rng = np.random.RandomState(11)
        n = 20
        X = rng.randn(n, 32).astype(np.float32)
        X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
        ids = [f"s{i}" for i in range(n)]
        lens = {s: 200 for s in ids}
        labels = np.arange(n)
        config = {
            "knn": {
                "mode": "rkcnn", "k": 3, "min_cosine": 0.0,
                "min_len_ratio": 0.0, "max_len_ratio": 100.0,
                "device": "cpu", "rkcnn_n_subspaces": 3,
                "rkcnn_subspace_fraction": 0.5, "rkcnn_n_neighbors": 3,
                "rkcnn_score_threshold": 0.0, "rkcnn_weighting": "uniform",
                "rkcnn_cascade_topn": 0, "rkcnn_random_state": 11,
            }
        }
        rows = rkcnn_candidate_edges(X, ids, lens, labels, config)
        for row in rows:
            assert -1.0 <= row["cosine"] <= 1.0

    def test_l2_normalization_enforced(self, tmp_path):
        """Embeddings that are NOT L2-normalized should be auto-normalized."""
        rng = np.random.RandomState(0)
        X = rng.randn(15, 16).astype(np.float32) * 10  # Not normalized
        ids = [f"s{i}" for i in range(15)]
        lens = {s: 200 for s in ids}
        labels = np.arange(15)
        config = {
            "knn": {
                "mode": "rkcnn", "k": 3, "min_cosine": 0.0,
                "min_len_ratio": 0.0, "max_len_ratio": 100.0,
                "device": "cpu", "rkcnn_n_subspaces": 3,
                "rkcnn_subspace_fraction": 0.5, "rkcnn_n_neighbors": 2,
                "rkcnn_score_threshold": 0.0, "rkcnn_weighting": "uniform",
                "rkcnn_cascade_topn": 0, "rkcnn_random_state": 0,
            }
        }
        # Should not raise — internally normalizes
        rows = rkcnn_candidate_edges(X, ids, lens, labels, config)
        assert isinstance(rows, list)


# ---------------------------------------------------------------------------
# Config validation tests
# ---------------------------------------------------------------------------

class TestConfigValidation:
    def test_default_config_valid(self):
        errors = validate_config(DEFAULT_CONFIG)
        assert errors == []

    def test_invalid_knn_mode(self):
        cfg = {**DEFAULT_CONFIG, "knn": {**DEFAULT_CONFIG["knn"], "mode": "invalid"}}
        errors = validate_config(cfg)
        assert any("knn.mode" in e for e in errors)

    def test_invalid_knn_device(self):
        cfg = {**DEFAULT_CONFIG, "knn": {**DEFAULT_CONFIG["knn"], "device": "tpu"}}
        errors = validate_config(cfg)
        assert any("knn.device" in e for e in errors)

    def test_invalid_rkcnn_weighting(self):
        cfg = {**DEFAULT_CONFIG, "knn": {**DEFAULT_CONFIG["knn"], "rkcnn_weighting": "magic"}}
        errors = validate_config(cfg)
        assert any("rkcnn_weighting" in e for e in errors)

    def test_rkcnn_subspace_fraction_range(self):
        cfg = {**DEFAULT_CONFIG, "knn": {**DEFAULT_CONFIG["knn"], "rkcnn_subspace_fraction": 2.0}}
        errors = validate_config(cfg)
        assert any("rkcnn_subspace_fraction" in e for e in errors)


# ---------------------------------------------------------------------------
# Pipeline integration: knn() with mode=rkcnn
# ---------------------------------------------------------------------------

class TestKnnRkcnnIntegration:
    def test_knn_rkcnn_mode(self, tmp_path):
        """Run the knn() pipeline function with mode=rkcnn and verify output."""
        from plm_cluster.pipeline import knn as knn_fn

        rng = np.random.RandomState(42)
        n, d = 15, 32
        X = rng.randn(n, d).astype(np.float32)
        ids = [f"subfam_{i:04d}" for i in range(n)]

        emb_path = tmp_path / "embeddings.npy"
        np.save(str(emb_path), X)

        ids_path = tmp_path / "ids.txt"
        ids_path.write_text("\n".join(ids) + "\n")

        import polars as pl
        lens_df = pl.DataFrame({
            "subfamily_id": ids,
            "rep_length_aa": [rng.randint(100, 500) for _ in ids],
        })
        lens_path = tmp_path / "lengths.tsv"
        lens_df.write_csv(str(lens_path), separator="\t")

        out_tsv = tmp_path / "edges.tsv"
        config = {
            "knn": {
                "mode": "rkcnn",
                "k": 5,
                "min_cosine": 0.0,
                "min_len_ratio": 0.0,
                "max_len_ratio": 100.0,
                "device": "cpu",
                "rkcnn_n_subspaces": 5,
                "rkcnn_subspace_fraction": 0.5,
                "rkcnn_n_neighbors": 3,
                "rkcnn_score_threshold": 0.0,
                "rkcnn_weighting": "separation",
                "rkcnn_cascade_topn": 10,
                "rkcnn_random_state": 42,
            }
        }

        logger = logging.getLogger("test_knn_rkcnn")
        knn_fn(str(emb_path), str(ids_path), str(lens_path), str(out_tsv),
               config, logger=logger)

        assert out_tsv.exists()
        result = pl.read_csv(str(out_tsv), separator="\t")
        expected_cols = {"q_subfamily_id", "t_subfamily_id", "cosine",
                         "q_len", "t_len", "len_ratio", "pass_len_ratio"}
        assert expected_cols.issubset(set(result.columns))

    def test_knn_standard_mode_unchanged(self, tmp_path):
        """Verify that mode=knn still works correctly (backward compat)."""
        from plm_cluster.pipeline import knn as knn_fn

        rng = np.random.RandomState(42)
        n, d = 10, 16
        X = rng.randn(n, d).astype(np.float32)
        ids = [f"subfam_{i:04d}" for i in range(n)]

        emb_path = tmp_path / "embeddings.npy"
        np.save(str(emb_path), X)

        ids_path = tmp_path / "ids.txt"
        ids_path.write_text("\n".join(ids) + "\n")

        import polars as pl
        lens_df = pl.DataFrame({
            "subfamily_id": ids,
            "rep_length_aa": [rng.randint(100, 500) for _ in ids],
        })
        lens_path = tmp_path / "lengths.tsv"
        lens_df.write_csv(str(lens_path), separator="\t")

        out_tsv = tmp_path / "edges.tsv"
        config = {
            "knn": {
                "mode": "knn",
                "k": 5,
                "min_cosine": 0.0,
                "min_len_ratio": 0.0,
                "max_len_ratio": 100.0,
                "device": "cpu",
            }
        }

        knn_fn(str(emb_path), str(ids_path), str(lens_path), str(out_tsv), config)

        assert out_tsv.exists()
        result = pl.read_csv(str(out_tsv), separator="\t")
        assert "q_subfamily_id" in result.columns
        assert "cosine" in result.columns

    def test_knn_resume_skips(self, tmp_path):
        """Resume should skip if output already exists."""
        from plm_cluster.pipeline import knn as knn_fn

        out_tsv = tmp_path / "edges.tsv"
        out_tsv.write_text("existing content")

        # Should not raise even with invalid paths
        knn_fn("nonexistent.npy", "nonexistent.txt", "nonexistent.tsv",
               str(out_tsv), DEFAULT_CONFIG, resume=True)
        assert out_tsv.read_text() == "existing content"
