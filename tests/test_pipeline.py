from pathlib import Path

import pandas as pd

from plm_cluster.config import load_config
from plm_cluster.pipeline import merge_graph, write_matrices


def test_merge_graph_union(tmp_path: Path):
    hmm = tmp_path / "hmm.tsv"
    emb = tmp_path / "emb.tsv"
    out = tmp_path / "merged.tsv"
    pd.DataFrame([
        {"q_subfamily_id": "s1", "t_subfamily_id": "s2", "edge_weight": 0.9},
    ]).to_csv(hmm, sep="\t", index=False)
    pd.DataFrame([
        {"q_subfamily_id": "s2", "t_subfamily_id": "s3", "cosine": 0.7},
    ]).to_csv(emb, sep="\t", index=False)
    cfg = load_config(None)
    merge_graph(str(hmm), str(emb), str(out), cfg)
    m = pd.read_csv(out, sep="\t")
    assert set(m["source"]) == {"hmm", "emb"}


def test_write_matrices(tmp_path: Path):
    sm = tmp_path / "subfamily_map.tsv"
    seg = tmp_path / "segments.tsv"
    out = tmp_path / "out"
    pd.DataFrame([
        {"protein_id": "p1", "subfamily_id": "s1", "is_rep": 1},
        {"protein_id": "p2", "subfamily_id": "s1", "is_rep": 0},
    ]).to_csv(sm, sep="\t", index=False)
    pd.DataFrame([
        {"protein_id": "p1", "family_id": "f1"},
        {"protein_id": "p2", "family_id": "f1"},
    ]).to_csv(seg, sep="\t", index=False)
    cfg = load_config(None)
    write_matrices(str(sm), str(seg), str(out), cfg)
    assert (out / "subfamily_x_protein_sparse.tsv").exists()
    assert (out / "family_x_protein_sparse.tsv").exists()
