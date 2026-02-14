from pathlib import Path

import numpy as np
import pandas as pd

from plm_cluster.config import load_config
from plm_cluster.pipeline import (
    build_profiles,
    cluster_families,
    hmm_hmm_edges,
    knn,
    map_proteins_to_families,
    merge_graph,
    mmseqs_cluster,
    write_matrices,
)


def test_smoke_pipeline_with_mocked_tools(tmp_path: Path, monkeypatch):
    cfg = load_config(None)
    proteins = tmp_path / "toy.faa"
    proteins.write_text(">p1\nMKTAYIAK\n>p2\nMKTAYIAK\n>p3\nGAVLILKK\n")

    def fake_require(tools, config_paths=None):
        return {t: t for t in tools}

    def fake_run(cmd, logger, cwd=None):
        if cmd[0] == "mmseqs" and cmd[1] == "createtsv":
            Path(cmd[-1]).write_text("p1\tp1\np1\tp2\np3\tp3\n")
        elif cmd[0] == "mmseqs" and cmd[1] == "result2flat":
            Path(cmd[-1]).write_text(">p1\nMKTAYIAK\n>p3\nGAVLILKK\n")
        elif cmd[0] == "mafft":
            fa = Path(cmd[-1])
            return fa.read_text()
        elif cmd[0] == "hhmake":
            out = Path(cmd[cmd.index("-o") + 1])
            out.write_text("HHsearch 3.3\nLENG 8\n")
        elif cmd[0] == "hhalign":
            out = Path(cmd[cmd.index("-o") + 1])
            out.write_text("Probab=98.0 E-value=1e-20 Score=200 Aligned_cols=8 Identities=75%\n")
        return ""

    monkeypatch.setattr("plm_cluster.pipeline.require_executables", fake_require)
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", fake_run)

    class DummyLogger:
        def info(self, *args, **kwargs):
            return None

    logger = DummyLogger()

    mmseqs_cluster(str(proteins), str(tmp_path / "results/01_mmseqs"), cfg, logger)
    build_profiles(
        str(proteins),
        str(tmp_path / "results/01_mmseqs/subfamily_map.tsv"),
        str(tmp_path / "results/02_profiles"),
        cfg,
        logger,
    )

    ids = ["subfam_000000", "subfam_000001"]
    emb = np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32)
    emb_dir = tmp_path / "results/04_embeddings"
    emb_dir.mkdir(parents=True)
    np.save(emb_dir / "embeddings.npy", emb)
    (emb_dir / "ids.txt").write_text("\n".join(ids) + "\n")
    pd.DataFrame({"subfamily_id": ids, "rep_length_aa": [8, 8]}).to_csv(emb_dir / "lengths.tsv", sep="\t", index=False)
    knn(str(emb_dir / "embeddings.npy"), str(emb_dir / "ids.txt"), str(emb_dir / "lengths.tsv"), str(emb_dir / "embedding_knn_edges.tsv"), cfg)

    hmm_hmm_edges(
        str(tmp_path / "results/02_profiles/subfamily_profile_index.tsv"),
        str(tmp_path / "results/03_hmm_hmm_edges"),
        cfg,
        logger,
        str(emb_dir / "embedding_knn_edges.tsv"),
    )

    merge_graph(
        str(tmp_path / "results/03_hmm_hmm_edges/hmm_hmm_edges_core.tsv"),
        str(emb_dir / "embedding_knn_edges.tsv"),
        str(tmp_path / "results/06_family_clustering/merged_edges_strict.tsv"),
        str(tmp_path / "results/06_family_clustering/merged_edges_functional.tsv"),
        cfg,
        str(tmp_path / "results/03_hmm_hmm_edges/hmm_hmm_edges_relaxed.tsv"),
    )

    cluster_families(
        str(tmp_path / "results/06_family_clustering/merged_edges_strict.tsv"),
        str(tmp_path / "results/06_family_clustering/merged_edges_functional.tsv"),
        str(tmp_path / "results/01_mmseqs/subfamily_map.tsv"),
        str(tmp_path / "results/06_family_clustering"),
        cfg,
    )

    map_proteins_to_families(
        str(proteins),
        str(tmp_path / "results/06_family_clustering/subfamily_to_family_strict.tsv"),
        str(tmp_path / "results/06_family_clustering/subfamily_to_family_functional.tsv"),
        str(tmp_path / "results/01_mmseqs/subfamily_map.tsv"),
        str(tmp_path / "results/05_domain_hits"),
        cfg,
    )

    write_matrices(
        str(tmp_path / "results/01_mmseqs/subfamily_map.tsv"),
        str(tmp_path / "results/05_domain_hits/protein_family_segments.tsv"),
        str(tmp_path / "results/07_membership_matrices"),
        cfg,
    )

    expected = [
        "results/01_mmseqs/subfamily_map.tsv",
        "results/02_profiles/subfamily_profile_index.tsv",
        "results/03_hmm_hmm_edges/hmm_hmm_edges_core.tsv",
        "results/03_hmm_hmm_edges/hmm_hmm_edges_relaxed.tsv",
        "results/04_embeddings/embedding_knn_edges.tsv",
        "results/06_family_clustering/subfamily_to_family_strict.tsv",
        "results/06_family_clustering/subfamily_to_family_functional.tsv",
        "results/05_domain_hits/protein_family_segments.tsv",
        "results/07_membership_matrices/family_strict_x_protein_sparse.tsv",
        "results/07_membership_matrices/family_functional_x_protein_sparse.tsv",
    ]
    for rel in expected:
        fp = tmp_path / rel
        assert fp.exists()
        assert fp.stat().st_size > 0
