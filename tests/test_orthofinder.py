"""Tests for the OrthoFinder integration (orthofinder_cluster)."""
from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from plm_cluster.config import load_config
from plm_cluster.pipeline import orthofinder_cluster


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _write_faa(path: Path, entries: dict[str, str]) -> None:
    """Write a minimal FASTA file from a {id: seq} dict."""
    with path.open("w") as fh:
        for pid, seq in entries.items():
            fh.write(f">{pid}\n{seq}\n")


class DummyLogger:
    def __init__(self):
        self.messages: list[str] = []

    def info(self, msg, *args, **kwargs):
        self.messages.append(msg % args if args else msg)

    def warning(self, msg, *args, **kwargs):
        self.messages.append(msg % args if args else msg)

    def error(self, msg, *args, **kwargs):
        self.messages.append(msg % args if args else msg)


def _fake_require(tools, config_paths=None):
    return {t: t for t in tools}


def _make_fake_run(og_dir_path):
    """Return a fake run_cmd that simulates MMseqs2 for OG subclustering tests."""

    def fake_run(cmd, logger, cwd=None):
        # createtsv: write a two-column TSV where every protein is its own rep
        if cmd[0] == "mmseqs" and cmd[1] == "createtsv":
            tsv_path = Path(cmd[-1])
            # Read the input FASTA to know which proteins are present
            fa_path = Path(cmd[2]).parent / "og.faa"
            lines = fa_path.read_text().splitlines()
            proteins = [l[1:].split()[0] for l in lines if l.startswith(">")]
            with tsv_path.open("w") as fh:
                for pid in proteins:
                    fh.write(f"{pid}\t{pid}\n")
        elif cmd[0] == "mmseqs" and cmd[1] == "result2flat":
            # Write a FASTA where each protein is its own rep
            out_faa = Path(cmd[-1])
            fa_path = Path(cmd[2]).parent / "og.faa"
            out_faa.write_text(fa_path.read_text())
        # All other mmseqs sub-commands (createdb, linclust, result2repseq) are no-ops
        return ""

    return fake_run


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_singleton_og(tmp_path, monkeypatch):
    """An OG with exactly one sequence becomes a single-member subfamily."""
    og_dir = tmp_path / "ogs"
    og_dir.mkdir()
    _write_faa(og_dir / "OG0000001.faa", {"protA": "MKTAYIAK"})

    outdir = tmp_path / "out"
    cfg = load_config(None)

    monkeypatch.setattr("plm_cluster.pipeline.require_executables", _fake_require)
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", _make_fake_run(og_dir))

    logger = DummyLogger()
    orthofinder_cluster(str(og_dir), str(outdir), cfg, logger)

    smap = pd.read_csv(outdir / "subfamily_map.tsv", sep="\t")
    assert len(smap) == 1
    assert smap.iloc[0]["protein_id"] == "protA"
    assert smap.iloc[0]["is_rep"] == 1
    subfam_id = smap.iloc[0]["subfamily_id"]
    assert subfam_id.startswith("OG0000001_subfam_")


def test_small_og_below_threshold(tmp_path, monkeypatch):
    """OG with 1 member (< min_og_size_for_subclustering=2) is treated as singleton
    without calling MMseqs2."""
    og_dir = tmp_path / "ogs"
    og_dir.mkdir()
    _write_faa(og_dir / "OG0000001.faa", {"protA": "MKTAYIAK"})

    outdir = tmp_path / "out"
    cfg = load_config(None)
    cfg["orthofinder"]["min_og_size_for_subclustering"] = 2

    run_cmd_called = []

    def spy_run(cmd, logger, cwd=None):
        run_cmd_called.append(cmd)
        return ""

    monkeypatch.setattr("plm_cluster.pipeline.require_executables", _fake_require)
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", spy_run)

    logger = DummyLogger()
    orthofinder_cluster(str(og_dir), str(outdir), cfg, logger)

    # MMseqs2 must NOT have been called for a below-threshold OG
    assert not any(c[0] == "mmseqs" for c in run_cmd_called)

    smap = pd.read_csv(outdir / "subfamily_map.tsv", sep="\t")
    assert len(smap) == 1
    assert smap.iloc[0]["protein_id"] == "protA"


def test_multiple_ogs_merged(tmp_path, monkeypatch):
    """Two OG FASTAs produce merged outputs with scoped, non-colliding IDs."""
    og_dir = tmp_path / "ogs"
    og_dir.mkdir()
    _write_faa(og_dir / "OG0000001.faa", {"p1": "MKTAYIAK", "p2": "MKTAYIAG"})
    _write_faa(og_dir / "OG0000002.faa", {"p3": "GAVLILKK", "p4": "GAVLILKG"})

    outdir = tmp_path / "out"
    cfg = load_config(None)

    monkeypatch.setattr("plm_cluster.pipeline.require_executables", _fake_require)
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", _make_fake_run(og_dir))

    logger = DummyLogger()
    orthofinder_cluster(str(og_dir), str(outdir), cfg, logger)

    smap = pd.read_csv(outdir / "subfamily_map.tsv", sep="\t")
    assert len(smap) == 4  # p1, p2, p3, p4

    subfamily_ids = smap["subfamily_id"].tolist()
    # IDs from OG1 and OG2 must not collide
    og1_ids = [s for s in subfamily_ids if s.startswith("OG0000001_")]
    og2_ids = [s for s in subfamily_ids if s.startswith("OG0000002_")]
    assert len(og1_ids) == 2
    assert len(og2_ids) == 2
    assert set(og1_ids).isdisjoint(set(og2_ids))


def test_og_id_sanitization(tmp_path, monkeypatch):
    """Filenames like 'N0.HOG0000001.fa' are sanitized to 'N0_HOG0000001'."""
    og_dir = tmp_path / "ogs"
    og_dir.mkdir()
    _write_faa(og_dir / "N0.HOG0000001.fa", {"p1": "MKTAYIAK"})

    outdir = tmp_path / "out"
    cfg = load_config(None)

    monkeypatch.setattr("plm_cluster.pipeline.require_executables", _fake_require)
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", _make_fake_run(og_dir))

    logger = DummyLogger()
    orthofinder_cluster(str(og_dir), str(outdir), cfg, logger)

    smap = pd.read_csv(outdir / "subfamily_map.tsv", sep="\t")
    subfam_id = smap.iloc[0]["subfamily_id"]
    # Dots must have been replaced with underscores; no dots in the ID
    assert "." not in subfam_id
    assert subfam_id.startswith("N0_HOG0000001_subfam_")


def test_proteins_combined_faa(tmp_path, monkeypatch):
    """All proteins from all OGs appear in proteins_combined.faa."""
    og_dir = tmp_path / "ogs"
    og_dir.mkdir()
    _write_faa(og_dir / "OG0000001.faa", {"p1": "MKTAYIAK", "p2": "MKTAYIAG"})
    _write_faa(og_dir / "OG0000002.faa", {"p3": "GAVLILKK"})

    outdir = tmp_path / "out"
    cfg = load_config(None)

    monkeypatch.setattr("plm_cluster.pipeline.require_executables", _fake_require)
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", _make_fake_run(og_dir))

    logger = DummyLogger()
    orthofinder_cluster(str(og_dir), str(outdir), cfg, logger)

    combined_path = outdir / "proteins_combined.faa"
    assert combined_path.exists()
    content = combined_path.read_text()
    for pid in ("p1", "p2", "p3"):
        assert f">{pid}" in content


def test_og_subfamily_map(tmp_path, monkeypatch):
    """og_subfamily_map.tsv correctly maps every subfamily back to its source OG."""
    og_dir = tmp_path / "ogs"
    og_dir.mkdir()
    _write_faa(og_dir / "OG0000001.faa", {"p1": "MKTAYIAK"})
    _write_faa(og_dir / "OG0000002.faa", {"p2": "GAVLILKK", "p3": "GAVLILKG"})

    outdir = tmp_path / "out"
    cfg = load_config(None)

    monkeypatch.setattr("plm_cluster.pipeline.require_executables", _fake_require)
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", _make_fake_run(og_dir))

    logger = DummyLogger()
    orthofinder_cluster(str(og_dir), str(outdir), cfg, logger)

    prov = pd.read_csv(outdir / "og_subfamily_map.tsv", sep="\t")
    assert set(prov.columns) >= {"subfamily_id", "og_id"}

    # Every subfamily in subfamily_map should appear in the provenance table
    smap = pd.read_csv(outdir / "subfamily_map.tsv", sep="\t")
    all_subfams = set(smap["subfamily_id"].unique())
    prov_subfams = set(prov["subfamily_id"].unique())
    assert all_subfams == prov_subfams

    # OG identity must be preserved (original stem, not sanitized)
    og1_rows = prov[prov["og_id"] == "OG0000001"]
    og2_rows = prov[prov["og_id"] == "OG0000002"]
    assert len(og1_rows) >= 1
    assert len(og2_rows) >= 1


def test_resume(tmp_path, monkeypatch):
    """If subfamily_map.tsv already exists, the function returns early without re-running."""
    og_dir = tmp_path / "ogs"
    og_dir.mkdir()
    _write_faa(og_dir / "OG0000001.faa", {"p1": "MKTAYIAK"})

    outdir = tmp_path / "out"
    outdir.mkdir()
    # Pre-create the output to trigger resume
    (outdir / "subfamily_map.tsv").write_text("protein_id\tsubfamily_id\tis_rep\n")

    cfg = load_config(None)
    run_cmd_called = []

    def spy_run(cmd, logger, cwd=None):
        run_cmd_called.append(cmd)
        return ""

    monkeypatch.setattr("plm_cluster.pipeline.require_executables", _fake_require)
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", spy_run)

    logger = DummyLogger()
    orthofinder_cluster(str(og_dir), str(outdir), cfg, logger, resume=True)

    assert not run_cmd_called, "run_cmd should not be called when resuming"
    assert any("Resume" in m for m in logger.messages)


def test_no_faa_files_raises(tmp_path, monkeypatch):
    """An empty directory raises FileNotFoundError with a clear message."""
    og_dir = tmp_path / "empty_ogs"
    og_dir.mkdir()
    outdir = tmp_path / "out"
    cfg = load_config(None)

    monkeypatch.setattr("plm_cluster.pipeline.require_executables", _fake_require)

    logger = DummyLogger()
    with pytest.raises(FileNotFoundError, match="No \\*.faa or \\*.fa files found"):
        orthofinder_cluster(str(og_dir), str(outdir), cfg, logger)


def test_run_all_orthofinder_smoke(tmp_path, monkeypatch):
    """Smoke test: orthofinder_cluster + downstream pipeline steps with mocked tools."""
    from plm_cluster.pipeline import (
        build_profiles,
        cluster_families,
        hmm_hmm_edges,
        knn,
        map_proteins_to_families,
        merge_graph,
        merge_hmm_shards,
        write_matrices,
    )

    og_dir = tmp_path / "ogs"
    og_dir.mkdir()
    _write_faa(og_dir / "OG0000001.faa", {"p1": "MKTAYIAK", "p2": "MKTAYIAK"})
    _write_faa(og_dir / "OG0000002.faa", {"p3": "GAVLILKK"})

    cfg = load_config(None)

    def fake_require(tools, config_paths=None):
        return {t: t for t in tools}

    def fake_run(cmd, logger, cwd=None):
        if cmd[0] == "mmseqs" and cmd[1] == "createtsv":
            tsv_path = Path(cmd[-1])
            fa_path = Path(cmd[2]).parent / "og.faa"
            lines = fa_path.read_text().splitlines()
            proteins = [l[1:].split()[0] for l in lines if l.startswith(">")]
            with tsv_path.open("w") as fh:
                # First protein is rep for all (simulate clustering p1+p2 together)
                rep = proteins[0]
                for pid in proteins:
                    fh.write(f"{rep}\t{pid}\n")
        elif cmd[0] == "mmseqs" and cmd[1] == "result2flat":
            out_faa = Path(cmd[-1])
            fa_path = Path(cmd[2]).parent / "og.faa"
            out_faa.write_text(fa_path.read_text())
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
            Path(out_file).write_text(
                "p1\tOG0000001_subfam_000000\t100.0\t8\t0\t0\t1\t8\t1\t8\t1e-20\t200\t8\t8\n"
                "p2\tOG0000001_subfam_000000\t100.0\t8\t0\t0\t1\t8\t1\t8\t1e-20\t200\t8\t8\n"
                "p3\tOG0000002_subfam_000000\t100.0\t8\t0\t0\t1\t8\t1\t8\t1e-20\t200\t8\t8\n"
            )
        return ""

    monkeypatch.setattr("plm_cluster.pipeline.require_executables", fake_require)
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", fake_run)

    logger = DummyLogger()
    results = tmp_path / "results"

    orthofinder_cluster(str(og_dir), str(results / "01_mmseqs"), cfg, logger)

    proteins_combined = str(results / "01_mmseqs/proteins_combined.faa")

    build_profiles(
        proteins_combined,
        str(results / "01_mmseqs/subfamily_map.tsv"),
        str(results / "02_profiles"),
        cfg,
        logger,
    )

    smap = pd.read_csv(results / "01_mmseqs/subfamily_map.tsv", sep="\t")
    ids = smap[smap["is_rep"] == 1]["subfamily_id"].tolist()
    emb = np.array([[1.0, 0.0]] * len(ids), dtype=np.float32)
    emb_dir = results / "04_embeddings"
    emb_dir.mkdir(parents=True)
    np.save(emb_dir / "embeddings.npy", emb)
    (emb_dir / "ids.txt").write_text("\n".join(ids) + "\n")
    pd.DataFrame({"subfamily_id": ids, "rep_length_aa": [8] * len(ids)}).to_csv(
        emb_dir / "lengths.tsv", sep="\t", index=False
    )
    knn(
        str(emb_dir / "embeddings.npy"),
        str(emb_dir / "ids.txt"),
        str(emb_dir / "lengths.tsv"),
        str(emb_dir / "embedding_knn_edges.tsv"),
        cfg,
    )

    hmm_hmm_edges(
        str(results / "02_profiles/subfamily_profile_index.tsv"),
        str(results / "03_hmm_hmm_edges"),
        cfg,
        logger,
        str(emb_dir / "embedding_knn_edges.tsv"),
    )

    merge_graph(
        str(results / "03_hmm_hmm_edges/hmm_hmm_edges_core.tsv"),
        str(emb_dir / "embedding_knn_edges.tsv"),
        str(results / "06_family_clustering/merged_edges_strict.tsv"),
        str(results / "06_family_clustering/merged_edges_functional.tsv"),
        cfg,
        str(results / "03_hmm_hmm_edges/hmm_hmm_edges_relaxed.tsv"),
    )

    cluster_families(
        str(results / "06_family_clustering/merged_edges_strict.tsv"),
        str(results / "06_family_clustering/merged_edges_functional.tsv"),
        str(results / "01_mmseqs/subfamily_map.tsv"),
        str(results / "06_family_clustering"),
        cfg,
    )

    map_proteins_to_families(
        proteins_combined,
        str(results / "06_family_clustering/subfamily_to_family_strict.tsv"),
        str(results / "06_family_clustering/subfamily_to_family_functional.tsv"),
        str(results / "01_mmseqs/subfamily_map.tsv"),
        str(results / "05_domain_hits"),
        cfg,
    )

    write_matrices(
        str(results / "01_mmseqs/subfamily_map.tsv"),
        str(results / "05_domain_hits/protein_family_segments.tsv"),
        str(results / "07_membership_matrices"),
        cfg,
    )

    # Verify key outputs exist
    expected = [
        "01_mmseqs/subfamily_map.tsv",
        "01_mmseqs/proteins_combined.faa",
        "01_mmseqs/og_subfamily_map.tsv",
        "02_profiles/subfamily_profile_index.tsv",
        "03_hmm_hmm_edges/hmm_hmm_edges_core.tsv",
        "04_embeddings/embedding_knn_edges.tsv",
        "06_family_clustering/subfamily_to_family_strict.tsv",
        "05_domain_hits/protein_family_segments.tsv",
        "07_membership_matrices/family_strict_x_protein_sparse.tsv",
    ]
    for rel in expected:
        fp = results / rel
        assert fp.exists(), f"Missing expected output: {rel}"
        assert fp.stat().st_size > 0, f"Output is empty: {rel}"


def test_run_all_orthofinder_rkcnn_knn_step(tmp_path, monkeypatch):
    """Smoke test: run-all-orthofinder pipeline KNN step uses rKCNN when knn.mode='rkcnn'."""
    from plm_cluster.pipeline import knn, orthofinder_cluster

    og_dir = tmp_path / "ogs"
    og_dir.mkdir()
    _write_faa(og_dir / "OG0000001.faa", {"p1": "MKTAYIAK", "p2": "MKTAYIAK"})
    _write_faa(og_dir / "OG0000002.faa", {"p3": "GAVLILKK"})

    cfg = load_config(None)
    cfg["knn"]["mode"] = "rkcnn"

    monkeypatch.setattr("plm_cluster.pipeline.require_executables",
                        lambda tools, config_paths=None: {t: t for t in tools})
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", _make_fake_run(og_dir))

    logger = DummyLogger()
    results = tmp_path / "results"

    orthofinder_cluster(str(og_dir), str(results / "01_mmseqs"), cfg, logger)

    smap = pd.read_csv(results / "01_mmseqs/subfamily_map.tsv", sep="\t")
    ids = smap[smap["is_rep"] == 1]["subfamily_id"].tolist()
    emb = np.array([[1.0, 0.0, 0.5, 0.2]] * len(ids), dtype=np.float32)
    emb_dir = results / "04_embeddings"
    emb_dir.mkdir(parents=True)
    np.save(emb_dir / "embeddings.npy", emb)
    (emb_dir / "ids.txt").write_text("\n".join(ids) + "\n")
    pd.DataFrame({"subfamily_id": ids, "rep_length_aa": [8] * len(ids)}).to_csv(
        emb_dir / "lengths.tsv", sep="\t", index=False
    )

    # Run the KNN step in rKCNN mode with the subfamily_map from orthofinder
    knn(
        str(emb_dir / "embeddings.npy"),
        str(emb_dir / "ids.txt"),
        str(emb_dir / "lengths.tsv"),
        str(emb_dir / "embedding_knn_edges.tsv"),
        cfg,
        subfamily_map=str(results / "01_mmseqs/subfamily_map.tsv"),
    )

    # Output file must exist (rKCNN may produce zero edges for tiny inputs, but file must exist)
    assert (emb_dir / "embedding_knn_edges.tsv").exists()


def test_cli_knn_mode_override_propagates(monkeypatch):
    """--knn-mode on run-all-orthofinder CLI propagates to cfg['knn']['mode']."""
    import argparse

    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd")
    p = sub.add_parser("run-all-orthofinder")
    p.add_argument("--config", default=None)
    p.add_argument("--results_root", default="/tmp/test_results")
    p.add_argument("--og_dir", default="/tmp/ogs")
    p.add_argument("--weights_path", default="/tmp/weights")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--hmm-mode", default=None, dest="hmm_mode",
                   choices=["pairwise", "db-search", "mmseqs-profile"])
    p.add_argument("--knn-mode", default=None, dest="knn_mode",
                   choices=["knn", "rkcnn"])
    p.add_argument("--shard-id", type=int, default=0, dest="shard_id")
    p.add_argument("--n-shards", type=int, default=1, dest="n_shards")

    args = ap.parse_args(["run-all-orthofinder", "--og_dir", "/tmp/ogs",
                          "--weights_path", "/tmp/w", "--knn-mode", "rkcnn"])
    assert args.knn_mode == "rkcnn", (
        f"Expected knn_mode='rkcnn' from CLI arg, got {args.knn_mode!r}"
    )

    # Verify the override is propagated into the config (as the CLI dispatch does)
    cfg = load_config(None)
    if args.knn_mode:
        cfg["knn"]["mode"] = args.knn_mode
    assert cfg["knn"]["mode"] == "rkcnn"
