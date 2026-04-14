from __future__ import annotations

from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


DEFAULT_CONFIG: dict[str, Any] = {
    "seed": 42,
    "tools": {
        "mmseqs_path": "",
        "hhmake_path": "",
        "hhsearch_path": "",
        "hhalign_path": "",
        "ffindex_build_path": "",
        "cstranslate_path": "",
        "mafft_path": "",
        "mcl_path": "",
    },
    "mmseqs": {
        "mode": "linclust",
        "min_seq_id": 0.6,
        "coverage": 0.8,
        "cov_mode": 1,
        "evalue": 1e-3,
        "sensitivity": 7.5,
        "threads": 8,
        "tmpdir": "tmp/mmseqs",
    },
    "profiles": {"max_members_per_subfamily": 256},
    "hmm_hmm": {
        "mode": "pairwise",
        "topN": 200,
        "mincov_core": 0.70,
        "min_prob_core": 90.0,
        "max_evalue_core": 1e-5,
        "min_aln_len_core": 120,
        "mincov_relaxed": 0.60,
        "min_prob_relaxed": 80.0,
        "max_evalue_relaxed": 1e-3,
        "min_aln_len_relaxed": 80,
    },
    "embed": {
        "esm_model_name": "esm2_t33_650M_UR50D",
        "batch_size": 4,
        "long_seq_policy": "truncate",
        "max_len": 2048,
        "pooling": "mean",
        "device": "cpu",
    },
    "knn": {
        "mode": "knn",
        "k": 100,
        "min_cosine": 0.35,
        "min_len_ratio": 0.5,
        "max_len_ratio": 2.0,
        "device": "cpu",
        # rKCNN-specific parameters (only used when mode="rkcnn")
        "rkcnn_n_subspaces": 50,
        "rkcnn_subspace_fraction": 0.5,
        "rkcnn_n_neighbors": 5,
        "rkcnn_score_threshold": 0.0,
        "rkcnn_weighting": "separation",
        "rkcnn_cascade_topn": 500,
        "rkcnn_random_state": 42,
    },
    "graph": {
        "edge_weight_policy": "downweight_embeddings",
        "w_hmm": 1.0,
        "w_emb": 0.35,
        "weak_hmm_threshold": 0.2,
        "leiden_resolution_strict": 0.7,
        "leiden_resolution_functional": 1.0,
        "seed": 42,
    },
    "mapping": {
        "search_mode": "family_reps",
        "profile_cov_min": 0.70,
        "min_prob": 90.0,
        "max_evalue": 1e-5,
        "min_segment_len": 120,
        "max_overlap_aa": 30,
    },
    "outputs": {"write_dense_threshold": 2_000_000, "write_matrix_market": True},
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
_RANGE_CHECKS: list[tuple[str, str, float, float]] = [
    ("mmseqs", "min_seq_id", 0.0, 1.0),
    ("mmseqs", "coverage", 0.0, 1.0),
    ("mmseqs", "evalue", 0.0, 1e6),
    ("mmseqs", "threads", 1, 1024),
    ("hmm_hmm", "mincov_core", 0.0, 1.0),
    ("hmm_hmm", "min_prob_core", 0.0, 100.0),
    ("hmm_hmm", "mincov_relaxed", 0.0, 1.0),
    ("hmm_hmm", "min_prob_relaxed", 0.0, 100.0),
    ("embed", "batch_size", 1, 512),
    ("embed", "max_len", 64, 100_000),
    ("knn", "k", 1, 100_000),
    ("knn", "min_cosine", -1.0, 1.0),
    ("knn", "rkcnn_n_subspaces", 1, 10_000),
    ("knn", "rkcnn_subspace_fraction", 0.01, 1.0),
    ("knn", "rkcnn_n_neighbors", 1, 10_000),
    ("knn", "rkcnn_score_threshold", 0.0, 1e6),
    ("knn", "rkcnn_cascade_topn", 0, 1_000_000),
    ("graph", "leiden_resolution_strict", 0.01, 100.0),
    ("graph", "leiden_resolution_functional", 0.01, 100.0),
    ("mapping", "profile_cov_min", 0.0, 1.0),
    ("mapping", "min_prob", 0.0, 100.0),
    ("mapping", "max_overlap_aa", 0, 10_000),
]

_VALID_EMBED_DEVICES = {"cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"}
_VALID_EDGE_POLICIES = {"strict", "union", "gated", "downweight_embeddings"}
_VALID_LONG_SEQ_POLICIES = {"truncate", "skip", "full"}
_VALID_HMM_MODES = {"pairwise", "db-search", "mmseqs-profile"}
_VALID_KNN_MODES = {"knn", "rkcnn"}
_VALID_RKCNN_WEIGHTINGS = {"separation", "uniform"}


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors (empty = OK)."""
    errors: list[str] = []
    for section, key, lo, hi in _RANGE_CHECKS:
        val = cfg.get(section, {}).get(key)
        if val is not None:
            try:
                v = float(val)
                if not (lo <= v <= hi):
                    errors.append(f"{section}.{key}={val} out of range [{lo}, {hi}]")
            except (ValueError, TypeError):
                errors.append(f"{section}.{key}={val!r} is not numeric")

    device = str(cfg.get("embed", {}).get("device", "cpu"))
    if device not in _VALID_EMBED_DEVICES and not device.startswith("cuda:"):
        errors.append(f"embed.device='{device}' is not valid; use 'cpu' or 'cuda[:N]'")

    policy = cfg.get("graph", {}).get("edge_weight_policy", "")
    if policy and policy not in _VALID_EDGE_POLICIES:
        errors.append(f"graph.edge_weight_policy='{policy}' not in {_VALID_EDGE_POLICIES}")

    lsp = cfg.get("embed", {}).get("long_seq_policy", "")
    if lsp and lsp not in _VALID_LONG_SEQ_POLICIES:
        errors.append(f"embed.long_seq_policy='{lsp}' not in {_VALID_LONG_SEQ_POLICIES}")

    hmm_mode = cfg.get("hmm_hmm", {}).get("mode", "")
    if hmm_mode and hmm_mode not in _VALID_HMM_MODES:
        errors.append(f"hmm_hmm.mode='{hmm_mode}' not in {_VALID_HMM_MODES}")

    knn_mode = cfg.get("knn", {}).get("mode", "")
    if knn_mode and knn_mode not in _VALID_KNN_MODES:
        errors.append(f"knn.mode='{knn_mode}' not in {_VALID_KNN_MODES}")

    knn_device = str(cfg.get("knn", {}).get("device", "cpu"))
    if knn_device not in _VALID_EMBED_DEVICES and not knn_device.startswith("cuda:"):
        errors.append(f"knn.device='{knn_device}' is not valid; use 'cpu' or 'cuda[:N]'")

    rkcnn_w = cfg.get("knn", {}).get("rkcnn_weighting", "")
    if rkcnn_w and rkcnn_w not in _VALID_RKCNN_WEIGHTINGS:
        errors.append(f"knn.rkcnn_weighting='{rkcnn_w}' not in {_VALID_RKCNN_WEIGHTINGS}")

    return errors


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None) -> dict[str, Any]:
    """Load config from a YAML file, deep-merged over defaults.

    Only ``.yaml`` and ``.yml`` files are accepted.  Passing a ``.json`` file
    raises a ``ValueError`` with a clear message.
    Validates parameter ranges and raises ``ValueError`` on problems.
    """
    if not path:
        return deepcopy(DEFAULT_CONFIG)
    p = Path(path)
    if p.suffix.lower() == ".json":
        raise ValueError(
            f"JSON config files are no longer supported: '{path}'. "
            "Please convert your configuration to YAML (.yaml or .yml)."
        )
    if p.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            f"Unsupported config file extension '{p.suffix}' for '{path}'. "
            "Only .yaml and .yml files are accepted."
        )
    with p.open() as f:
        user = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULT_CONFIG, user)
    errors = validate_config(cfg)
    if errors:
        raise ValueError("Config validation errors:\n  " + "\n  ".join(errors))
    return cfg

