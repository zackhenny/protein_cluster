from pathlib import Path

import pandas as pd

from plm_cluster.config import load_config
from plm_cluster.pipeline import merge_graph, write_matrices


def test_merge_graph_writes_strict_and_functional(tmp_path: Path):
    hmm_core = tmp_path / "hmm_core.tsv"
    hmm_relaxed = tmp_path / "hmm_relaxed.tsv"
    emb = tmp_path / "emb.tsv"
    out_s = tmp_path / "merged_s.tsv"
    out_f = tmp_path / "merged_f.tsv"

    pd.DataFrame([
        {"q_subfamily_id": "s1", "t_subfamily_id": "s2", "edge_weight": 0.9},
    ]).to_csv(hmm_core, sep="\t", index=False)
    pd.DataFrame([
        {"q_subfamily_id": "s2", "t_subfamily_id": "s3", "edge_weight": 0.4},
    ]).to_csv(hmm_relaxed, sep="\t", index=False)
    pd.DataFrame([
        {"q_subfamily_id": "s3", "t_subfamily_id": "s4", "cosine": 0.7},
    ]).to_csv(emb, sep="\t", index=False)

    cfg = load_config(None)
    merge_graph(str(hmm_core), str(emb), str(out_s), str(out_f), cfg, str(hmm_relaxed))

    assert out_s.exists()
    assert out_f.exists()
    strict = pd.read_csv(out_s, sep="\t")
    functional = pd.read_csv(out_f, sep="\t")
    assert len(strict) == 1
    assert {"hmm_core", "hmm_relaxed", "emb"}.intersection(set(functional["source"]))


def test_write_matrices_family_modes(tmp_path: Path):
    sm = tmp_path / "subfamily_map.tsv"
    seg = tmp_path / "segments.tsv"
    out = tmp_path / "out"

    pd.DataFrame([
        {"protein_id": "p1", "subfamily_id": "s1", "is_rep": 1},
        {"protein_id": "p2", "subfamily_id": "s1", "is_rep": 0},
    ]).to_csv(sm, sep="\t", index=False)

    pd.DataFrame([
        {"protein_id": "p1", "family_id": "famS_000001", "family_mode": "strict"},
        {"protein_id": "p1", "family_id": "famF_000001", "family_mode": "functional"},
        {"protein_id": "p2", "family_id": "famS_000001", "family_mode": "strict"},
    ]).to_csv(seg, sep="\t", index=False)

    write_matrices(str(sm), str(seg), str(out), load_config(None))
    assert (out / "subfamily_x_protein_sparse.tsv").exists()
    assert (out / "family_strict_x_protein_sparse.tsv").exists()
    assert (out / "family_functional_x_protein_sparse.tsv").exists()
