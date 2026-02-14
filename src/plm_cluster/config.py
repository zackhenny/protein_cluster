from __future__ import annotations

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
    "hmm_hmm": {"topN": 200, "mincov": 0.7, "min_prob": 90, "max_evalue": 1e-5, "min_aln_len": 120},
    "embed": {"esm_model_name": "esm2_t33_650M_UR50D", "batch_size": 4, "long_seq_policy": "truncate", "max_len": 2048},
    "knn": {"k": 100, "min_cosine": 0.35, "min_len_ratio": 0.5, "max_len_ratio": 2.0},
    "graph": {"merge_policy": "union", "w_hmm": 1.0, "w_emb": 0.5, "leiden_resolution": 0.8, "seed": 42},
    "mapping": {"profile_cov_min": 0.7, "min_prob": 90, "max_evalue": 1e-5, "min_segment_len": 120},
    "outputs": {"write_dense_threshold": 2000000, "write_matrix_market": True},
}


def load_config(path: str | None) -> dict[str, Any]:
    if not path:
        return DEFAULT_CONFIG.copy()
    with Path(path).open() as f:
        user = yaml.safe_load(f) or {}
    merged = DEFAULT_CONFIG.copy()
    for k, v in user.items():
        if isinstance(v, dict) and isinstance(merged.get(k), dict):
            merged[k] = {**merged[k], **v}
        else:
            merged[k] = v
    return merged
