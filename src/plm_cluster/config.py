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


def _deep_merge(base: dict[str, Any], updates: dict[str, Any]) -> dict[str, Any]:
    out = deepcopy(base)
    for k, v in updates.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return deepcopy(DEFAULT_CONFIG)
    with Path(path).open() as f:
        user = yaml.safe_load(f) or {}
    return _deep_merge(DEFAULT_CONFIG, user)
