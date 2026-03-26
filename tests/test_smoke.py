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
    merge_hmm_shards,
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
        elif cmd[0] == "mmseqs" and cmd[1] == "easy-search":
            out_file = cmd[4]
            Path(out_file).write_text("p1\tsubfam_000000\t100.0\t8\t0\t0\t1\t8\t1\t8\t1e-20\t200\t8\t8\n"
                                      "p2\tsubfam_000000\t100.0\t8\t0\t0\t1\t8\t1\t8\t1e-20\t200\t8\t8\n"
                                      "p3\tsubfam_000001\t100.0\t8\t0\t0\t1\t8\t1\t8\t1e-20\t200\t8\t8\n")
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

    # Progress log should have been created
    progress = tmp_path / "results/03_hmm_hmm_edges/hmm_hmm_progress.ndjson"
    assert progress.exists()


def _make_db_search_fixtures(tmp_path: Path):
    """Create the profile index and embeddings needed for HMM-HMM tests."""
    profile_dir = tmp_path / "profiles"
    profile_dir.mkdir(parents=True)
    ids = ["subfam_000000", "subfam_000001"]
    hhm_paths = {}
    for sid in ids:
        p = profile_dir / f"{sid}.hhm"
        p.write_text(f"HHsearch 3.3\nNAME {sid}\nLENG 8\n")
        hhm_paths[sid] = str(p)

    idx_path = tmp_path / "subfamily_profile_index.tsv"
    pd.DataFrame({"subfamily_id": ids, "hhm_path": [hhm_paths[s] for s in ids]}).to_csv(
        idx_path, sep="\t", index=False
    )

    emb_dir = tmp_path / "embeddings"
    emb_dir.mkdir(parents=True)
    emb = np.array([[1.0, 0.0], [0.8, 0.2]], dtype=np.float32)
    np.save(emb_dir / "embeddings.npy", emb)
    (emb_dir / "ids.txt").write_text("\n".join(ids) + "\n")
    pd.DataFrame({"subfamily_id": ids, "rep_length_aa": [8, 8]}).to_csv(
        emb_dir / "lengths.tsv", sep="\t", index=False
    )
    knn(
        str(emb_dir / "embeddings.npy"),
        str(emb_dir / "ids.txt"),
        str(emb_dir / "lengths.tsv"),
        str(emb_dir / "embedding_knn_edges.tsv"),
        load_config(None),
    )
    return str(idx_path), str(emb_dir / "embedding_knn_edges.tsv")


def test_hmm_hmm_edges_db_search_mode(tmp_path: Path, monkeypatch):
    """db-search mode produces the same output files as pairwise with mocked tools."""
    cfg = load_config(None)
    profile_index, candidate_edges = _make_db_search_fixtures(tmp_path)
    outdir = str(tmp_path / "out")

    def fake_require(tools, config_paths=None):
        return {t: t for t in tools}

    def fake_run_db(cmd, logger, cwd=None):
        if cmd[0] == "ffindex_build":
            # Create the ffdata and ffindex files
            Path(cmd[1]).write_text("dummy_ffdata")
            Path(cmd[2]).write_text("dummy_ffindex")
        elif cmd[0] == "hhsearch":
            out_path = Path(cmd[cmd.index("-o") + 1])
            # Write a minimal .hhr output with one hit
            other = "subfam_000001" if "subfam_000000" in cmd[cmd.index("-i") + 1] else "subfam_000000"
            out_path.write_text(
                f"No 1\n>{other} description\n"
                f"Probab=98.0 E-value=1e-20 Score=200.0 Aligned_cols=8 Identities=75%\n"
            )
        return ""

    monkeypatch.setattr("plm_cluster.pipeline.require_executables", fake_require)
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", fake_run_db)

    class DummyLogger:
        def info(self, *a, **kw): pass
        def warning(self, *a, **kw): pass
        def error(self, *a, **kw): pass

    hmm_hmm_edges(profile_index, outdir, cfg, DummyLogger(), candidate_edges, mode="db-search")

    out = Path(outdir)
    assert (out / "hmm_hmm_edges_raw.tsv").exists()
    assert (out / "hmm_hmm_edges_core.tsv").exists()
    assert (out / "hmm_hmm_edges_relaxed.tsv").exists()
    assert (out / "hmm_hmm_progress.ndjson").exists()

    # Verify progress log has valid NDJSON entries
    import json as _json
    progress_lines = [
        _json.loads(l) for l in (out / "hmm_hmm_progress.ndjson").read_text().splitlines() if l.strip()
    ]
    assert all("q" in r and "t" in r and "status" in r for r in progress_lines)


def test_hmm_hmm_edges_resume(tmp_path: Path, monkeypatch):
    """--resume skips already-completed pairs recorded in the progress log."""
    cfg = load_config(None)
    profile_index, candidate_edges = _make_db_search_fixtures(tmp_path)
    outdir = str(tmp_path / "out")
    Path(outdir).mkdir(parents=True)

    call_count = {"hhalign": 0}

    def fake_require(tools, config_paths=None):
        return {t: t for t in tools}

    def fake_run_resume(cmd, logger, cwd=None):
        if cmd[0] == "hhalign":
            call_count["hhalign"] += 1
            out_path = Path(cmd[cmd.index("-o") + 1])
            out_path.write_text("Probab=98.0 E-value=1e-20 Score=200.0 Aligned_cols=8 Identities=75%\n")
        return ""

    monkeypatch.setattr("plm_cluster.pipeline.require_executables", fake_require)
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", fake_run_resume)

    class DummyLogger:
        def info(self, *a, **kw): pass
        def warning(self, *a, **kw): pass
        def error(self, *a, **kw): pass

    # First run — should call hhalign
    hmm_hmm_edges(profile_index, outdir, cfg, DummyLogger(), candidate_edges)
    first_count = call_count["hhalign"]
    assert first_count > 0

    # Second run with --resume — should call hhalign 0 additional times (all pairs done)
    hmm_hmm_edges(profile_index, outdir, cfg, DummyLogger(), candidate_edges, resume=True)
    assert call_count["hhalign"] == first_count  # no new calls


def test_hmm_hmm_edges_sharding(tmp_path: Path, monkeypatch):
    """Sharding splits candidate pairs and merge_hmm_shards reassembles them."""
    cfg = load_config(None)
    profile_index, candidate_edges = _make_db_search_fixtures(tmp_path)
    outdir = str(tmp_path / "out")

    def fake_require(tools, config_paths=None):
        return {t: t for t in tools}

    def fake_run_shard(cmd, logger, cwd=None):
        if cmd[0] == "hhalign":
            out_path = Path(cmd[cmd.index("-o") + 1])
            out_path.write_text("Probab=95.0 E-value=1e-10 Score=150.0 Aligned_cols=8 Identities=70%\n")
        return ""

    monkeypatch.setattr("plm_cluster.pipeline.require_executables", fake_require)
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", fake_run_shard)

    class DummyLogger:
        def info(self, *a, **kw): pass
        def warning(self, *a, **kw): pass
        def error(self, *a, **kw): pass

    logger = DummyLogger()
    n_shards = 2
    for shard_id in range(n_shards):
        hmm_hmm_edges(
            profile_index, outdir, cfg, logger, candidate_edges,
            shard_id=shard_id, n_shards=n_shards,
        )

    out = Path(outdir)
    assert (out / "hmm_hmm_edges_raw.shard_0.tsv").exists()
    assert (out / "hmm_hmm_edges_raw.shard_1.tsv").exists()

    # Merge shards
    merge_hmm_shards(outdir, cfg, logger)
    assert (out / "hmm_hmm_edges_raw.tsv").exists()
    assert (out / "hmm_hmm_edges_core.tsv").exists()
    assert (out / "hmm_hmm_edges_relaxed.tsv").exists()
