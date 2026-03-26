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
    "knn": {"k": 100, "min_cosine": 0.35, "min_len_ratio": 0.5, "max_len_ratio": 2.0},
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
    ("graph", "leiden_resolution_strict", 0.01, 100.0),
    ("graph", "leiden_resolution_functional", 0.01, 100.0),
    ("mapping", "profile_cov_min", 0.0, 1.0),
    ("mapping", "min_prob", 0.0, 100.0),
    ("mapping", "max_overlap_aa", 0, 10_000),
]

_VALID_EMBED_DEVICES = {"cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"}
_VALID_EDGE_POLICIES = {"strict", "union", "gated", "downweight_embeddings"}
_VALID_LONG_SEQ_POLICIES = {"truncate", "skip", "full"}


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
    """Load config from a YAML or JSON file, deep-merged over defaults.

    File format is detected by extension: ``.json`` → JSON, anything else → YAML.
    Validates parameter ranges and raises ``ValueError`` on problems.
    """
    if not path:
        return deepcopy(DEFAULT_CONFIG)
    p = Path(path)
    with p.open() as f:
        if p.suffix.lower() == ".json":
            import json
            user = json.load(f) or {}
        else:
            user = yaml.safe_load(f) or {}
    cfg = _deep_merge(DEFAULT_CONFIG, user)
    errors = validate_config(cfg)
    if errors:
        raise ValueError("Config validation errors:\n  " + "\n  ".join(errors))
    return cfg

