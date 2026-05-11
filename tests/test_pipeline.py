import logging
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from plm_cluster.config import load_config
from plm_cluster.pipeline import _build_hhsuite_db, build_profiles, embed, merge_graph, mmseqs_cluster, write_matrices


def test_load_config_rejects_json(tmp_path: Path):
    cfg_json = tmp_path / "config.json"
    cfg_json.write_text('{"seed": 1}')
    with pytest.raises(ValueError, match="JSON config files are no longer supported"):
        load_config(str(cfg_json))


def test_load_config_rejects_unsupported_extension(tmp_path: Path):
    cfg_txt = tmp_path / "config.toml"
    cfg_txt.write_text("seed = 1")
    with pytest.raises(ValueError, match="Unsupported config file extension"):
        load_config(str(cfg_txt))


def test_load_config_accepts_yaml(tmp_path: Path):
    cfg_yaml = tmp_path / "config.yaml"
    cfg_yaml.write_text("seed: 99\n")
    cfg = load_config(str(cfg_yaml))
    assert cfg["seed"] == 99


def test_load_config_accepts_yml(tmp_path: Path):
    cfg_yml = tmp_path / "config.yml"
    cfg_yml.write_text("seed: 77\n")
    cfg = load_config(str(cfg_yml))
    assert cfg["seed"] == 77


def test_merge_graph_writes_strict_and_functional(tmp_path: Path):
    hmm_core = tmp_path / "hmm_core.tsv"
    hmm_relaxed = tmp_path / "hmm_relaxed.tsv"
    emb = tmp_path / "emb.tsv"
    out_s = tmp_path / "nested/06_family_clustering/merged_s.tsv"
    out_f = tmp_path / "nested/06_family_clustering/merged_f.tsv"

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


def test_build_hhsuite_db_uses_prefix_and_builds_cs219(tmp_path: Path):
    """_build_hhsuite_db should use db_prefix for file paths and build the _cs219 database."""
    db_dir = tmp_path / "hhsearch_db"
    db_prefix = db_dir / "mydb"

    # Create fake .hhm files so the file-list is non-empty
    hhm1 = tmp_path / "subfam_000001.hhm"
    hhm1.write_text("HHM content")
    hhm_paths = {"subfam_000001": str(hhm1)}

    # No a3m files → empty _a3m branch
    a3m_paths = {}

    logger = logging.getLogger("test_hhsuite_db")

    commands_run: list[list] = []

    def fake_run_cmd(cmd, _logger):
        commands_run.append(cmd)
        # Simulate ffindex_build creating the output files
        if "ffindex_build" in cmd[0]:
            # cmd: [bin, -s, -f, listfile, ffdata, ffindex]
            Path(cmd[-2]).write_bytes(b"dummy")
            Path(cmd[-1]).write_text("dummy")
        # Simulate cstranslate creating the cs219 files
        if "cstranslate" in cmd[0]:
            # -o argument is the cs219 prefix; create .ffdata and .ffindex
            o_idx = cmd.index("-o") + 1
            cs219_prefix = cmd[o_idx]
            Path(cs219_prefix + ".ffdata").write_bytes(b"cs219_data")
            Path(cs219_prefix + ".ffindex").write_text("cs219_idx")

    with patch("plm_cluster.pipeline.run_cmd", side_effect=fake_run_cmd):
        _build_hhsuite_db(
            db_dir, db_prefix, hhm_paths, a3m_paths,
            ffindex_build_bin="ffindex_build",
            cstranslate_bin="cstranslate",
            logger=logger,
        )

    # Verify _hhm files are named after db_prefix (not a hardcoded 'profiles_db')
    assert (db_dir / "mydb_hhm.ffdata").exists()
    assert (db_dir / "mydb_hhm.ffindex").exists()

    # Verify _a3m empty placeholders are also under db_prefix
    assert (db_dir / "mydb_a3m.ffdata").exists()
    assert (db_dir / "mydb_a3m.ffindex").exists()

    # Verify _cs219 database was created
    assert (db_dir / "mydb_cs219.ffdata").exists()
    assert (db_dir / "mydb_cs219.ffindex").exists()

    # Verify cstranslate was called with correct arguments
    cstranslate_calls = [c for c in commands_run if "cstranslate" in c[0]]
    assert len(cstranslate_calls) == 1
    cs_cmd = cstranslate_calls[0]
    assert "-i" in cs_cmd and str(db_prefix) + "_a3m" in cs_cmd
    assert "-o" in cs_cmd and str(db_prefix) + "_cs219" in cs_cmd
    assert "-b" in cs_cmd


def test_build_hhsuite_db_skips_existing_cs219(tmp_path: Path):
    """_build_hhsuite_db should skip cstranslate when _cs219 files already exist."""
    db_dir = tmp_path / "hhsearch_db"
    db_dir.mkdir()
    db_prefix = db_dir / "mydb"

    # Pre-create all six database files
    for suffix in ("_hhm.ffdata", "_hhm.ffindex", "_a3m.ffdata", "_a3m.ffindex",
                   "_cs219.ffdata", "_cs219.ffindex"):
        Path(str(db_prefix) + suffix).write_bytes(b"existing")

    hhm_paths = {"s1": str(tmp_path / "s1.hhm")}
    logger = logging.getLogger("test_hhsuite_db_skip")

    commands_run: list[list] = []

    with patch("plm_cluster.pipeline.run_cmd", side_effect=lambda c, l: commands_run.append(c)):
        _build_hhsuite_db(
            db_dir, db_prefix, hhm_paths, {},
            ffindex_build_bin="ffindex_build",
            cstranslate_bin="cstranslate",
            logger=logger,
        )

    # No commands should have been run because all files already exist
    assert commands_run == []



# ---------------------------------------------------------------------------
# cluster_families(): singleton subfamily preservation
# ---------------------------------------------------------------------------

def _write_edges(path: Path, rows):
    """Write a minimal edges TSV."""
    import pandas as pd
    pd.DataFrame(rows).to_csv(path, sep="\t", index=False)


def test_cluster_families_singleton_preservation(tmp_path: Path):
    """Subfamilies with no graph edges must appear in both output TSVs as singletons."""
    from plm_cluster.pipeline import cluster_families

    # Three subfamilies: s1-s2 are connected, s3 has no edges at all
    _write_edges(tmp_path / "strict.tsv", [
        {"qid": "s1", "tid": "s2", "weight": 0.9, "source": "hmm_core"},
    ])
    _write_edges(tmp_path / "func.tsv", [
        {"qid": "s1", "tid": "s2", "weight": 0.8, "source": "hmm_core"},
    ])

    smap_path = tmp_path / "subfamily_map.tsv"
    pd.DataFrame([
        {"protein_id": "p1", "subfamily_id": "s1", "is_rep": 1},
        {"protein_id": "p2", "subfamily_id": "s2", "is_rep": 1},
        {"protein_id": "p3", "subfamily_id": "s3", "is_rep": 1},
    ]).to_csv(smap_path, sep="\t", index=False)

    out_dir = tmp_path / "out"
    cfg = load_config(None)

    cluster_families(
        str(tmp_path / "strict.tsv"),
        str(tmp_path / "func.tsv"),
        str(smap_path),
        str(out_dir),
        cfg,
        logger=logging.getLogger("test"),
    )

    strict = pd.read_csv(out_dir / "subfamily_to_family_strict.tsv", sep="\t")
    func = pd.read_csv(out_dir / "subfamily_to_family_functional.tsv", sep="\t")

    for df, name in [(strict, "strict"), (func, "functional")]:
        assigned = set(df["subfamily_id"].tolist())
        assert "s3" in assigned, f"s3 missing from {name} output (no graph edges)"
        assert "s1" in assigned, f"s1 missing from {name} output"
        assert "s2" in assigned, f"s2 missing from {name} output"


def test_cluster_families_singleton_gets_unique_family_id(tmp_path: Path):
    """Each edge-less subfamily must receive a distinct singleton family ID."""
    from plm_cluster.pipeline import cluster_families

    # s1 and s2 are connected; s3 and s4 are isolated
    _write_edges(tmp_path / "strict.tsv", [
        {"qid": "s1", "tid": "s2", "weight": 0.9, "source": "hmm_core"},
    ])
    _write_edges(tmp_path / "func.tsv", [
        {"qid": "s1", "tid": "s2", "weight": 0.8, "source": "hmm_core"},
    ])

    smap_path = tmp_path / "subfamily_map.tsv"
    pd.DataFrame([
        {"protein_id": "p1", "subfamily_id": "s1", "is_rep": 1},
        {"protein_id": "p2", "subfamily_id": "s2", "is_rep": 1},
        {"protein_id": "p3", "subfamily_id": "s3", "is_rep": 1},
        {"protein_id": "p4", "subfamily_id": "s4", "is_rep": 1},
    ]).to_csv(smap_path, sep="\t", index=False)

    out_dir = tmp_path / "out"
    cfg = load_config(None)
    cluster_families(
        str(tmp_path / "strict.tsv"),
        str(tmp_path / "func.tsv"),
        str(smap_path),
        str(out_dir),
        cfg,
        logger=logging.getLogger("test"),
    )

    strict = pd.read_csv(out_dir / "subfamily_to_family_strict.tsv", sep="\t")
    # The two isolated subfamilies (s3, s4) must have DIFFERENT family IDs
    s3_fam = strict.loc[strict["subfamily_id"] == "s3", "family_id"].iloc[0]
    s4_fam = strict.loc[strict["subfamily_id"] == "s4", "family_id"].iloc[0]
    assert s3_fam != s4_fam, "s3 and s4 should have different singleton family IDs"


def test_cluster_families_all_isolated_subfamilies(tmp_path: Path):
    """When ALL subfamilies are edge-less (empty graphs), every subfamily still appears."""
    from plm_cluster.pipeline import cluster_families

    # Empty edge graphs
    _write_edges(tmp_path / "strict.tsv", [
        {"qid": "dummy", "tid": "dummy", "weight": 0.5, "source": "x"},
    ])
    _write_edges(tmp_path / "func.tsv", [
        {"qid": "dummy", "tid": "dummy", "weight": 0.5, "source": "x"},
    ])

    # All proteins in their own isolated subfamilies (none in graphs above)
    smap_path = tmp_path / "subfamily_map.tsv"
    pd.DataFrame([
        {"protein_id": "p1", "subfamily_id": "A", "is_rep": 1},
        {"protein_id": "p2", "subfamily_id": "B", "is_rep": 1},
        {"protein_id": "p3", "subfamily_id": "C", "is_rep": 1},
    ]).to_csv(smap_path, sep="\t", index=False)

    out_dir = tmp_path / "out"
    cfg = load_config(None)
    cluster_families(
        str(tmp_path / "strict.tsv"),
        str(tmp_path / "func.tsv"),
        str(smap_path),
        str(out_dir),
        cfg,
        logger=logging.getLogger("test"),
    )

    strict = pd.read_csv(out_dir / "subfamily_to_family_strict.tsv", sep="\t")
    assigned = set(strict["subfamily_id"].tolist())
    assert assigned >= {"A", "B", "C"}, f"Not all subfamilies assigned: {assigned}"


# ---------------------------------------------------------------------------
# mapping.min_prob backward-compatibility
# ---------------------------------------------------------------------------

def test_load_config_min_prob_deprecated_alias(tmp_path: Path):
    """loading a config with mapping.min_prob emits a DeprecationWarning and maps to min_pident."""
    import warnings

    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text("mapping:\n  min_prob: 50.0\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(str(cfg_yaml))

    dep_warns = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert dep_warns, "Expected a DeprecationWarning for mapping.min_prob"
    assert cfg["mapping"]["min_pident"] == 50.0


def test_load_config_min_pident_canonical(tmp_path: Path):
    """mapping.min_pident is accepted without any warning."""
    import warnings

    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text("mapping:\n  min_pident: 30.0\n")

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        cfg = load_config(str(cfg_yaml))

    dep_warns = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert not dep_warns, "No DeprecationWarning expected for min_pident"
    assert cfg["mapping"]["min_pident"] == 30.0


# ---------------------------------------------------------------------------
# OrthoFinder subcluster mode selection (cluster | linclust | auto)
# ---------------------------------------------------------------------------

def _write_faa_file(path: Path, entries: dict[str, str]) -> None:
    with path.open("w") as fh:
        for pid, seq in entries.items():
            fh.write(f">{pid}\n{seq}\n")


def _make_fake_run_record(_og_dir_path, recorded_cmds):
    """Return a fake run_cmd that records all commands issued."""
    def fake_run(cmd, logger, cwd=None):
        recorded_cmds.append(list(cmd))
        if cmd[0] == "mmseqs" and cmd[1] == "createtsv":
            tsv_path = Path(cmd[-1])
            fa_path = Path(cmd[2]).parent / "og.faa"
            lines = fa_path.read_text().splitlines()
            proteins = [l[1:].split()[0] for l in lines if l.startswith(">")]
            with tsv_path.open("w") as fh:
                for pid in proteins:
                    fh.write(f"{pid}\t{pid}\n")
        elif cmd[0] == "mmseqs" and cmd[1] == "result2flat":
            out_faa = Path(cmd[-1])
            fa_path = Path(cmd[2]).parent / "og.faa"
            out_faa.write_text(fa_path.read_text())
        return ""
    return fake_run


def test_orthofinder_linclust_mode(tmp_path, monkeypatch):
    """subcluster_mode='linclust' must call 'mmseqs linclust', never 'mmseqs cluster'."""
    og_dir = tmp_path / "ogs"
    og_dir.mkdir()
    _write_faa_file(og_dir / "OG0000001.faa", {"p1": "MKTAYIAK", "p2": "MKTAYIAG"})

    cfg = load_config(None)
    cfg["orthofinder"]["subcluster_mode"] = "linclust"

    cmds: list[list] = []
    monkeypatch.setattr("plm_cluster.pipeline.require_executables", lambda t, c=None: {k: k for k in t})
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", _make_fake_run_record(og_dir, cmds))

    from plm_cluster.pipeline import orthofinder_cluster
    orthofinder_cluster(str(og_dir), str(tmp_path / "out"), cfg, logging.getLogger("test"))

    algo_calls = [c[1] for c in cmds if c[0] == "mmseqs" and len(c) > 1]
    assert "linclust" in algo_calls
    assert "cluster" not in algo_calls


def test_orthofinder_cluster_mode(tmp_path, monkeypatch):
    """subcluster_mode='cluster' must call 'mmseqs cluster', never 'mmseqs linclust'."""
    og_dir = tmp_path / "ogs"
    og_dir.mkdir()
    _write_faa_file(og_dir / "OG0000001.faa", {"p1": "MKTAYIAK", "p2": "MKTAYIAG"})

    cfg = load_config(None)
    cfg["orthofinder"]["subcluster_mode"] = "cluster"

    cmds: list[list] = []
    monkeypatch.setattr("plm_cluster.pipeline.require_executables", lambda t, c=None: {k: k for k in t})
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", _make_fake_run_record(og_dir, cmds))

    from plm_cluster.pipeline import orthofinder_cluster
    orthofinder_cluster(str(og_dir), str(tmp_path / "out"), cfg, logging.getLogger("test"))

    algo_calls = [c[1] for c in cmds if c[0] == "mmseqs" and len(c) > 1]
    assert "cluster" in algo_calls
    assert "linclust" not in algo_calls


def test_orthofinder_auto_mode_small_og_uses_cluster(tmp_path, monkeypatch):
    """auto mode with a small OG (below threshold) must use mmseqs cluster."""
    og_dir = tmp_path / "ogs"
    og_dir.mkdir()
    # 3 members, threshold = 1000 → should use "cluster"
    _write_faa_file(og_dir / "OG0000001.faa", {"p1": "MKTAYIAK", "p2": "MKTAYIAG", "p3": "GAVLILKK"})

    cfg = load_config(None)
    cfg["orthofinder"]["subcluster_mode"] = "auto"
    cfg["orthofinder"]["auto_linclust_min_size"] = 1000

    cmds: list[list] = []
    monkeypatch.setattr("plm_cluster.pipeline.require_executables", lambda t, c=None: {k: k for k in t})
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", _make_fake_run_record(og_dir, cmds))

    from plm_cluster.pipeline import orthofinder_cluster
    orthofinder_cluster(str(og_dir), str(tmp_path / "out"), cfg, logging.getLogger("test"))

    algo_calls = [c[1] for c in cmds if c[0] == "mmseqs" and len(c) > 1]
    assert "cluster" in algo_calls
    assert "linclust" not in algo_calls


def test_orthofinder_auto_mode_large_og_uses_linclust(tmp_path, monkeypatch):
    """auto mode with a large OG (at or above threshold) must use mmseqs linclust."""
    og_dir = tmp_path / "ogs"
    og_dir.mkdir()
    # 3 members, threshold = 2 → should use "linclust"
    _write_faa_file(og_dir / "OG0000001.faa", {"p1": "MKTAYIAK", "p2": "MKTAYIAG", "p3": "GAVLILKK"})

    cfg = load_config(None)
    cfg["orthofinder"]["subcluster_mode"] = "auto"
    cfg["orthofinder"]["auto_linclust_min_size"] = 2  # threshold=2, OG has 3 → linclust

    cmds: list[list] = []
    monkeypatch.setattr("plm_cluster.pipeline.require_executables", lambda t, c=None: {k: k for k in t})
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", _make_fake_run_record(og_dir, cmds))

    from plm_cluster.pipeline import orthofinder_cluster
    orthofinder_cluster(str(og_dir), str(tmp_path / "out"), cfg, logging.getLogger("test"))

    algo_calls = [c[1] for c in cmds if c[0] == "mmseqs" and len(c) > 1]
    assert "linclust" in algo_calls
    assert "cluster" not in algo_calls


def test_orthofinder_cluster_reassign_passed(tmp_path, monkeypatch):
    """subcluster_cluster_reassign=True must pass --cluster-reassign to mmseqs cluster."""
    og_dir = tmp_path / "ogs"
    og_dir.mkdir()
    _write_faa_file(og_dir / "OG0000001.faa", {"p1": "MKTAYIAK", "p2": "MKTAYIAG"})

    cfg = load_config(None)
    cfg["orthofinder"]["subcluster_mode"] = "cluster"
    cfg["orthofinder"]["subcluster_cluster_reassign"] = True

    cmds: list[list] = []
    monkeypatch.setattr("plm_cluster.pipeline.require_executables", lambda t, c=None: {k: k for k in t})
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", _make_fake_run_record(og_dir, cmds))

    from plm_cluster.pipeline import orthofinder_cluster
    orthofinder_cluster(str(og_dir), str(tmp_path / "out"), cfg, logging.getLogger("test"))

    cluster_cmds = [c for c in cmds if c[0] == "mmseqs" and len(c) > 1 and c[1] == "cluster"]
    assert cluster_cmds, "Expected at least one 'mmseqs cluster' call"
    assert any("--cluster-reassign" in c for c in cluster_cmds), \
        "--cluster-reassign not found in cluster command"


def test_orthofinder_cluster_reassign_not_passed_for_linclust(tmp_path, monkeypatch):
    """subcluster_cluster_reassign must NOT be passed when mode is linclust."""
    og_dir = tmp_path / "ogs"
    og_dir.mkdir()
    _write_faa_file(og_dir / "OG0000001.faa", {"p1": "MKTAYIAK", "p2": "MKTAYIAG"})

    cfg = load_config(None)
    cfg["orthofinder"]["subcluster_mode"] = "linclust"
    cfg["orthofinder"]["subcluster_cluster_reassign"] = True  # set but must be ignored

    cmds: list[list] = []
    monkeypatch.setattr("plm_cluster.pipeline.require_executables", lambda t, c=None: {k: k for k in t})
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", _make_fake_run_record(og_dir, cmds))

    from plm_cluster.pipeline import orthofinder_cluster
    orthofinder_cluster(str(og_dir), str(tmp_path / "out"), cfg, logging.getLogger("test"))

    linclust_cmds = [c for c in cmds if c[0] == "mmseqs" and len(c) > 1 and c[1] == "linclust"]
    assert linclust_cmds, "Expected at least one 'mmseqs linclust' call"
    for cmd in linclust_cmds:
        assert "--cluster-reassign" not in cmd, \
            "--cluster-reassign should NOT appear in linclust command"


# ---------------------------------------------------------------------------
# CLI: --hmm-mode precedence over config
# ---------------------------------------------------------------------------

def test_config_orthofinder_subcluster_mode_validation():
    """Invalid orthofinder.subcluster_mode should raise ValueError."""
    from plm_cluster.config import load_config
    import tempfile, os
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        f.write("orthofinder:\n  subcluster_mode: invalid_mode\n")
        tmp = f.name
    try:
        with pytest.raises(ValueError, match="subcluster_mode"):
            load_config(tmp)
    finally:
        os.unlink(tmp)


def test_config_new_orthofinder_keys_loaded(tmp_path: Path):
    """New orthofinder config keys should round-trip through load_config."""
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        "orthofinder:\n"
        "  subcluster_mode: cluster\n"
        "  auto_linclust_min_size: 500\n"
        "  subcluster_alignment_mode: 3\n"
        "  subcluster_cluster_reassign: true\n"
        "  subcluster_threads: 2\n"
        "  parallel_og_workers: 2\n"
    )
    cfg = load_config(str(cfg_yaml))
    of = cfg["orthofinder"]
    assert of["subcluster_mode"] == "cluster"
    assert of["auto_linclust_min_size"] == 500
    assert of["subcluster_alignment_mode"] == 3
    assert of["subcluster_cluster_reassign"] is True
    assert of["subcluster_threads"] == 2
    assert of["parallel_og_workers"] == 2


def test_config_parallel_workers_loaded(tmp_path: Path):
    """profiles.parallel_workers and hmm_hmm.parallel_workers should load correctly."""
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text(
        "profiles:\n  parallel_workers: 16\n"
        "hmm_hmm:\n  parallel_workers: 12\n"
    )
    cfg = load_config(str(cfg_yaml))
    assert cfg["profiles"]["parallel_workers"] == 16
    assert cfg["hmm_hmm"]["parallel_workers"] == 12


def _make_fake_esm_torch(embed_dim: int = 4):
    """Return fake (esm, torch) module objects suitable for sys.modules injection."""
    import types
    import unittest.mock as um

    # ---- torch ----
    torch_mod = types.ModuleType("torch")

    class _FakeDevice:
        def __init__(self, s):
            self.type = s.split(":")[0]
        def __str__(self):
            return self.type

    class _NoGrad:
        def __enter__(self): return self
        def __exit__(self, *a): pass

    torch_mod.device = _FakeDevice
    torch_mod.no_grad = lambda: _NoGrad()
    torch_mod.load = um.MagicMock(return_value={})
    torch_mod.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None)

    class _FakeTensor:
        def __init__(self, arr):
            self._d = np.asarray(arr, dtype=np.float32)
            self.shape = self._d.shape
        def to(self, device): return self
        def __getitem__(self, idx): return _FakeTensor(self._d[idx])
        def mean(self, dim): return _FakeTensor(self._d.mean(axis=dim))
        def cpu(self): return self
        def numpy(self): return self._d

    class _FakeModel:
        num_layers = 1
        def eval(self): return self
        def to(self, device): return self
        def __call__(self, toks, repr_layers, return_contacts):
            batch_sz = toks.shape[0]
            return {"representations": {1: _FakeTensor(
                np.random.rand(batch_sz, 20, embed_dim).astype(np.float32)
            )}}

    class _FakeAlphabet:
        def get_batch_converter(self):
            def _conv(batch):
                labels = [b[0] for b in batch]
                strs_ = [b[1] for b in batch]
                max_l = max(len(s) for s in strs_) if strs_ else 0
                toks = _FakeTensor(np.zeros((len(batch), max_l + 2)))
                return labels, strs_, toks
            return _conv

    # ---- esm ----
    esm_mod = types.ModuleType("esm")
    esm_mod.pretrained = types.SimpleNamespace(
        load_model_and_alphabet_local=lambda p: (_FakeModel(), _FakeAlphabet())
    )

    # functools.partial(torch_mod.load, weights_only=False) must be callable
    torch_mod.load = um.MagicMock(return_value={})

    return esm_mod, torch_mod


def _run_embed(tmp_path, fasta_text, cfg_overrides=None, weights="fake.pt"):
    """Write fasta to tmp_path, run embed() with mocked ESM/torch."""
    import sys, unittest.mock as um

    fasta = tmp_path / "reps.faa"
    fasta.write_text(fasta_text)
    weights_file = tmp_path / weights
    weights_file.write_text("dummy")

    cfg = load_config(None)
    if cfg_overrides:
        for section, key, val in cfg_overrides:
            cfg[section][key] = val

    esm_mod, torch_mod = _make_fake_esm_torch()
    log = logging.getLogger("test_embed")

    with um.patch.dict(sys.modules, {"esm": esm_mod, "torch": torch_mod}):
        embed(str(fasta), str(tmp_path / "out"), cfg, str(weights_file), log)

    return tmp_path / "out"


def test_embed_skips_empty_sequences(tmp_path: Path):
    """Sequences that become empty after stripping stop codons are skipped."""
    fasta_text = ">empty_seq\n*\n>normal_seq\nMKTAYIAK\n"
    out = _run_embed(tmp_path, fasta_text)

    ids = [x for x in (out / "ids.txt").read_text().splitlines() if x]
    assert "normal_seq" in ids
    assert "empty_seq" not in ids
    mat = np.load(out / "embeddings.npy")
    assert mat.shape[0] == 1


def test_embed_skip_policy_excludes_long_seqs(tmp_path: Path):
    """long_seq_policy='skip' must exclude sequences longer than max_len."""
    short = "MKTAYIAK"    # 8 AA
    long_seq = "A" * 20  # 20 AA > max_len=10
    fasta_text = f">short_seq\n{short}\n>long_seq\n{long_seq}\n"

    out = _run_embed(
        tmp_path, fasta_text,
        cfg_overrides=[("embed", "long_seq_policy", "skip"), ("embed", "max_len", 10)],
    )

    ids = [x for x in (out / "ids.txt").read_text().splitlines() if x]
    assert "short_seq" in ids
    assert "long_seq" not in ids
    mat = np.load(out / "embeddings.npy")
    assert mat.shape[0] == 1


def test_embed_empty_fasta_raises(tmp_path: Path):
    """embed() must raise RuntimeError when the FASTA has no embeddable sequences."""
    import sys, unittest.mock as um
    fasta_text = ">only_stop\n*\n"
    fasta = tmp_path / "reps.faa"
    fasta.write_text(fasta_text)
    weights_file = tmp_path / "fake.pt"
    weights_file.write_text("dummy")

    cfg = load_config(None)
    log = logging.getLogger("test_embed")
    esm_mod, torch_mod = _make_fake_esm_torch()

    with um.patch.dict(sys.modules, {"esm": esm_mod, "torch": torch_mod}):
        with pytest.raises(RuntimeError, match="No sequences remain to embed"):
            embed(str(fasta), str(tmp_path / "out"), cfg, str(weights_file), log)


# ---------------------------------------------------------------------------
# mmseqs_cluster(): min_protein_length pre-filtering
# ---------------------------------------------------------------------------

class _DummyLogger:
    def info(self, *a, **kw): pass
    def warning(self, *a, **kw): pass
    def error(self, *a, **kw): pass


def _make_mmseqs_fake_run(tsv_content: str, faa_content: str = ">p1\nMKTAYIAK\n"):
    """Return a fake run_cmd for mmseqs_cluster tests."""
    def fake_run(cmd, logger, cwd=None, stdin=None):
        if cmd[0] == "mmseqs" and cmd[1] == "createtsv":
            Path(cmd[-1]).write_text(tsv_content)
        elif cmd[0] == "mmseqs" and cmd[1] == "result2flat":
            Path(cmd[-1]).write_text(faa_content)
        # align / convertalis / createdb / linclust / result2repseq: no-ops
        return ""
    return fake_run


def test_mmseqs_cluster_min_protein_length_filters_short(tmp_path: Path, monkeypatch):
    """Proteins shorter than min_protein_length are removed before clustering."""
    cfg = load_config(None)
    cfg["mmseqs"]["min_protein_length"] = 10

    proteins = tmp_path / "toy.faa"
    # "short" is 5 AA, "long" is 15 AA
    proteins.write_text(">short\nMKTAY\n>long\nMKTAYIAKTAYIAKT\n")

    monkeypatch.setattr(
        "plm_cluster.pipeline.require_executables",
        lambda t, c=None: {k: k for k in t},
    )
    monkeypatch.setattr(
        "plm_cluster.pipeline.run_cmd",
        _make_mmseqs_fake_run("long\tlong\n", ">long\nMKTAYIAKTAYIAKT\n"),
    )

    outdir = tmp_path / "out"
    mmseqs_cluster(str(proteins), str(outdir), cfg, _DummyLogger())

    filtered = pd.read_csv(outdir / "filtered_short_proteins.tsv", sep="\t")
    assert list(filtered["protein_id"]) == ["short"]
    assert filtered["length_aa"].iloc[0] == 5

    smap = pd.read_csv(outdir / "subfamily_map.tsv", sep="\t")
    assert set(smap["protein_id"]) == {"long"}

    report = pd.read_csv(outdir / "mmseqs_cluster_report.tsv", sep="\t")
    assert report["n_proteins_input"].iloc[0] == 2
    assert report["n_proteins_filtered_short"].iloc[0] == 1
    assert report["n_proteins_clustered"].iloc[0] == 1


def test_mmseqs_cluster_min_protein_length_zero_no_filter(tmp_path: Path, monkeypatch):
    """With min_protein_length=0 (default) no proteins are filtered."""
    cfg = load_config(None)
    cfg["mmseqs"]["min_protein_length"] = 0

    proteins = tmp_path / "toy.faa"
    proteins.write_text(">p1\nMK\n>p2\nMKTAYIAK\n")

    monkeypatch.setattr(
        "plm_cluster.pipeline.require_executables",
        lambda t, c=None: {k: k for k in t},
    )
    monkeypatch.setattr(
        "plm_cluster.pipeline.run_cmd",
        _make_mmseqs_fake_run("p1\tp1\np2\tp2\n", ">p1\nMK\n>p2\nMKTAYIAK\n"),
    )

    outdir = tmp_path / "out"
    mmseqs_cluster(str(proteins), str(outdir), cfg, _DummyLogger())

    assert not (outdir / "filtered_short_proteins.tsv").exists()

    report = pd.read_csv(outdir / "mmseqs_cluster_report.tsv", sep="\t")
    assert report["n_proteins_filtered_short"].iloc[0] == 0


def test_mmseqs_cluster_report_singleton_counts(tmp_path: Path, monkeypatch):
    """mmseqs_cluster_report.tsv must correctly count singletons vs multi-member clusters."""
    cfg = load_config(None)
    cfg["mmseqs"]["min_cluster_size_for_profile"] = 2

    proteins = tmp_path / "toy.faa"
    proteins.write_text(">p1\nMKTAYIAK\n>p2\nMKTAYIAK\n>p3\nGAVLILKK\n")

    monkeypatch.setattr(
        "plm_cluster.pipeline.require_executables",
        lambda t, c=None: {k: k for k in t},
    )
    monkeypatch.setattr(
        "plm_cluster.pipeline.run_cmd",
        _make_mmseqs_fake_run("p1\tp1\np1\tp2\np3\tp3\n", ">p1\nMKTAYIAK\n>p3\nGAVLILKK\n"),
    )

    outdir = tmp_path / "out"
    mmseqs_cluster(str(proteins), str(outdir), cfg, _DummyLogger())

    report = pd.read_csv(outdir / "mmseqs_cluster_report.tsv", sep="\t")
    assert report["n_clusters_total"].iloc[0] == 2
    assert report["n_singletons"].iloc[0] == 1
    assert report["n_clusters_2plus"].iloc[0] == 1
    assert report["n_clusters_above_min_profile_size"].iloc[0] == 1


def test_mmseqs_cluster_stats_is_singleton_column(tmp_path: Path, monkeypatch):
    """subfamily_stats.tsv must have is_singleton=1 for singletons, 0 otherwise."""
    cfg = load_config(None)

    proteins = tmp_path / "toy.faa"
    proteins.write_text(">p1\nMKTAYIAK\n>p2\nMKTAYIAK\n>p3\nGAVLILKK\n")

    monkeypatch.setattr(
        "plm_cluster.pipeline.require_executables",
        lambda t, c=None: {k: k for k in t},
    )
    monkeypatch.setattr(
        "plm_cluster.pipeline.run_cmd",
        _make_mmseqs_fake_run("p1\tp1\np1\tp2\np3\tp3\n", ">p1\nMKTAYIAK\n>p3\nGAVLILKK\n"),
    )

    outdir = tmp_path / "out"
    mmseqs_cluster(str(proteins), str(outdir), cfg, _DummyLogger())

    stats = pd.read_csv(outdir / "subfamily_stats.tsv", sep="\t")
    assert "is_singleton" in stats.columns
    assert len(stats[stats["is_singleton"] == 1]) == 1
    assert len(stats[stats["is_singleton"] == 0]) == 1


def test_mmseqs_cluster_stats_length_columns(tmp_path: Path, monkeypatch):
    """subfamily_stats.tsv must include min/max/mean/std_length_aa columns."""
    cfg = load_config(None)
    proteins = tmp_path / "toy.faa"
    proteins.write_text(">p1\nMKTAYIAK\n>p2\nMKTAYIAK\n>p3\nGAVLILKK\n")

    monkeypatch.setattr(
        "plm_cluster.pipeline.require_executables",
        lambda t, c=None: {k: k for k in t},
    )
    monkeypatch.setattr(
        "plm_cluster.pipeline.run_cmd",
        _make_mmseqs_fake_run("p1\tp1\np1\tp2\np3\tp3\n", ">p1\nMKTAYIAK\n>p3\nGAVLILKK\n"),
    )

    outdir = tmp_path / "out"
    mmseqs_cluster(str(proteins), str(outdir), cfg, _DummyLogger())

    stats = pd.read_csv(outdir / "subfamily_stats.tsv", sep="\t")
    for col in ("min_length_aa", "max_length_aa", "mean_length_aa", "std_length_aa"):
        assert col in stats.columns, f"Expected column {col} in subfamily_stats.tsv"
    multi = stats[stats["n_members"] == 2].iloc[0]
    assert multi["min_length_aa"] == 8
    assert multi["max_length_aa"] == 8
    assert multi["std_length_aa"] == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# build_profiles(): min_cluster_size_for_profile skip logic
# ---------------------------------------------------------------------------

def test_build_profiles_skips_singletons_when_min_size_2(tmp_path: Path, monkeypatch):
    """build_profiles skips singletons when min_cluster_size_for_profile=2."""
    cfg = load_config(None)
    cfg["mmseqs"]["min_cluster_size_for_profile"] = 2

    proteins = tmp_path / "toy.faa"
    proteins.write_text(">p1\nMKTAYIAK\n>p2\nMKTAYIAK\n>p3\nGAVLILKK\n")

    smap = tmp_path / "smap.tsv"
    pd.DataFrame([
        {"protein_id": "p1", "subfamily_id": "subfam_000000", "is_rep": 1},
        {"protein_id": "p2", "subfamily_id": "subfam_000000", "is_rep": 0},
        {"protein_id": "p3", "subfamily_id": "subfam_000001", "is_rep": 1},
    ]).to_csv(smap, sep="\t", index=False)

    hhmake_calls: list[str] = []

    def fake_run(cmd, logger, cwd=None, stdin=None):
        if cmd[0] == "mafft":
            return Path(cmd[-1]).read_text()
        if cmd[0] == "hhmake":
            name_idx = cmd.index("-name")
            hhmake_calls.append(cmd[name_idx + 1])
            out_path = Path(cmd[cmd.index("-o") + 1])
            out_path.write_text(f"HHsearch 3.3\nNAME {cmd[name_idx + 1]}\nLENG 8\n")
        return ""

    monkeypatch.setattr(
        "plm_cluster.pipeline.require_executables",
        lambda t, c=None: {k: k for k in t},
    )
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", fake_run)

    outdir = tmp_path / "profiles"
    outdir.mkdir()
    build_profiles(str(proteins), str(smap), str(outdir), cfg, _DummyLogger())

    assert hhmake_calls == ["subfam_000000"]

    idx = pd.read_csv(outdir / "subfamily_profile_index.tsv", sep="\t")
    assert len(idx) == 2
    assert "profile_built" in idx.columns
    assert "skipped_reason" in idx.columns

    skipped = idx[idx["subfamily_id"] == "subfam_000001"].iloc[0]
    assert not skipped["profile_built"]
    assert skipped["skipped_reason"] == "singleton"


def test_build_profiles_skips_below_min_cluster_size(tmp_path: Path, monkeypatch):
    """build_profiles also skips clusters with 2 members when threshold=3."""
    cfg = load_config(None)
    cfg["mmseqs"]["min_cluster_size_for_profile"] = 3

    proteins = tmp_path / "toy.faa"
    proteins.write_text(">p1\nMKTAYIAK\n>p2\nMKTAYIAK\n>p3\nGAVLILKK\n")

    smap = tmp_path / "smap.tsv"
    pd.DataFrame([
        {"protein_id": "p1", "subfamily_id": "subfam_000000", "is_rep": 1},
        {"protein_id": "p2", "subfamily_id": "subfam_000000", "is_rep": 0},
        {"protein_id": "p3", "subfamily_id": "subfam_000001", "is_rep": 1},
    ]).to_csv(smap, sep="\t", index=False)

    hhmake_calls: list[str] = []

    def fake_run(cmd, logger, cwd=None, stdin=None):
        if cmd[0] == "mafft":
            return Path(cmd[-1]).read_text()
        if cmd[0] == "hhmake":
            name_idx = cmd.index("-name")
            hhmake_calls.append(cmd[name_idx + 1])
            out_path = Path(cmd[cmd.index("-o") + 1])
            out_path.write_text("HHsearch 3.3\nLENG 8\n")
        return ""

    monkeypatch.setattr(
        "plm_cluster.pipeline.require_executables",
        lambda t, c=None: {k: k for k in t},
    )
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", fake_run)

    outdir = tmp_path / "profiles"
    outdir.mkdir()
    build_profiles(str(proteins), str(smap), str(outdir), cfg, _DummyLogger())

    assert hhmake_calls == []

    idx = pd.read_csv(outdir / "subfamily_profile_index.tsv", sep="\t")
    assert len(idx) == 2
    assert not any(idx["profile_built"])
    singleton = idx[idx["subfamily_id"] == "subfam_000001"].iloc[0]
    two_member = idx[idx["subfamily_id"] == "subfam_000000"].iloc[0]
    assert singleton["skipped_reason"] == "singleton"
    assert two_member["skipped_reason"] == "below_min_cluster_size"


def test_build_profiles_default_builds_all(tmp_path: Path, monkeypatch):
    """With default min_cluster_size_for_profile=1, singletons still get profiles."""
    cfg = load_config(None)
    assert cfg["mmseqs"]["min_cluster_size_for_profile"] == 1

    proteins = tmp_path / "toy.faa"
    proteins.write_text(">p1\nMKTAYIAK\n>p2\nGAVLILKK\n")

    smap = tmp_path / "smap.tsv"
    pd.DataFrame([
        {"protein_id": "p1", "subfamily_id": "subfam_000000", "is_rep": 1},
        {"protein_id": "p2", "subfamily_id": "subfam_000001", "is_rep": 1},
    ]).to_csv(smap, sep="\t", index=False)

    hhmake_calls: list[str] = []

    def fake_run(cmd, logger, cwd=None, stdin=None):
        if cmd[0] == "mafft":
            return Path(cmd[-1]).read_text()
        if cmd[0] == "hhmake":
            name_idx = cmd.index("-name")
            hhmake_calls.append(cmd[name_idx + 1])
            out_path = Path(cmd[cmd.index("-o") + 1])
            out_path.write_text("HHsearch 3.3\nLENG 8\n")
        return ""

    monkeypatch.setattr(
        "plm_cluster.pipeline.require_executables",
        lambda t, c=None: {k: k for k in t},
    )
    monkeypatch.setattr("plm_cluster.pipeline.run_cmd", fake_run)

    outdir = tmp_path / "profiles"
    outdir.mkdir()
    build_profiles(str(proteins), str(smap), str(outdir), cfg, _DummyLogger())

    assert sorted(hhmake_calls) == ["subfam_000000", "subfam_000001"]
    idx = pd.read_csv(outdir / "subfamily_profile_index.tsv", sep="\t")
    assert all(idx["profile_built"])


# ---------------------------------------------------------------------------
# Config validation: new keys
# ---------------------------------------------------------------------------

def test_config_min_protein_length_loaded(tmp_path: Path):
    """mmseqs.min_protein_length is accepted and round-trips through load_config."""
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text("mmseqs:\n  min_protein_length: 50\n")
    cfg = load_config(str(cfg_yaml))
    assert cfg["mmseqs"]["min_protein_length"] == 50


def test_config_min_protein_length_range_validation(tmp_path: Path):
    """mmseqs.min_protein_length must be >= 0; negative values are rejected."""
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text("mmseqs:\n  min_protein_length: -1\n")
    with pytest.raises(ValueError, match="min_protein_length"):
        load_config(str(cfg_yaml))


def test_config_min_cluster_size_for_profile_loaded(tmp_path: Path):
    """mmseqs.min_cluster_size_for_profile is accepted and round-trips through load_config."""
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text("mmseqs:\n  min_cluster_size_for_profile: 5\n")
    cfg = load_config(str(cfg_yaml))
    assert cfg["mmseqs"]["min_cluster_size_for_profile"] == 5


def test_config_min_cluster_size_for_profile_range_validation(tmp_path: Path):
    """mmseqs.min_cluster_size_for_profile must be >= 1; 0 is rejected."""
    cfg_yaml = tmp_path / "cfg.yaml"
    cfg_yaml.write_text("mmseqs:\n  min_cluster_size_for_profile: 0\n")
    with pytest.raises(ValueError, match="min_cluster_size_for_profile"):
        load_config(str(cfg_yaml))


# ---------------------------------------------------------------------------
# QC plots: new functions render without error
# ---------------------------------------------------------------------------

def test_qc_new_plots_render_with_empty_data(tmp_path: Path):
    """New QC plot functions handle missing data gracefully (set_visible(False))."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    from plm_cluster.qc_plots import (
        plot_cluster_identity_range,
        plot_cluster_length_variation,
        plot_singleton_summary,
    )

    results_root = str(tmp_path)
    for func in (plot_singleton_summary, plot_cluster_length_variation, plot_cluster_identity_range):
        fig, ax = plt.subplots()
        func(results_root, ax)  # must not raise
        plt.close(fig)


def test_qc_singleton_summary_with_report(tmp_path: Path):
    """plot_singleton_summary reads mmseqs_cluster_report.tsv correctly."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    from plm_cluster.qc_plots import plot_singleton_summary

    mm_dir = tmp_path / "01_mmseqs"
    mm_dir.mkdir(parents=True)
    pd.DataFrame({
        "n_proteins_input": [100],
        "n_proteins_filtered_short": [5],
        "n_proteins_clustered": [95],
        "n_clusters_total": [50],
        "n_singletons": [20],
        "n_clusters_2plus": [30],
        "n_clusters_above_min_profile_size": [30],
    }).to_csv(mm_dir / "mmseqs_cluster_report.tsv", sep="\t", index=False)
    pd.DataFrame({
        "subfamily_id": [f"s{i}" for i in range(50)],
        "n_members": [1] * 20 + [2] * 15 + [4] * 15,
    }).to_csv(mm_dir / "subfamily_stats.tsv", sep="\t", index=False)

    fig, ax = plt.subplots()
    plot_singleton_summary(str(tmp_path), ax)  # must not raise
    plt.close(fig)


def test_qc_cluster_length_variation_with_data(tmp_path: Path):
    """plot_cluster_length_variation renders a scatter for multi-member clusters."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    from plm_cluster.qc_plots import plot_cluster_length_variation

    mm_dir = tmp_path / "01_mmseqs"
    mm_dir.mkdir(parents=True)
    pd.DataFrame({
        "subfamily_id": ["s1", "s2", "s3"],
        "n_members": [1, 5, 10],
        "std_length_aa": [0.0, 15.0, 25.0],
    }).to_csv(mm_dir / "subfamily_stats.tsv", sep="\t", index=False)

    fig, ax = plt.subplots()
    plot_cluster_length_variation(str(tmp_path), ax)
    plt.close(fig)


def test_qc_cluster_identity_range_with_data(tmp_path: Path):
    """plot_cluster_identity_range renders a scatter for multi-member clusters."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        pytest.skip("matplotlib not installed")

    from plm_cluster.qc_plots import plot_cluster_identity_range

    mm_dir = tmp_path / "01_mmseqs"
    mm_dir.mkdir(parents=True)
    pd.DataFrame({
        "subfamily_id": ["s1", "s2"],
        "n_members": [5, 10],
        "min_pident": [70.0, 60.0],
        "max_pident": [100.0, 95.0],
    }).to_csv(mm_dir / "subfamily_stats.tsv", sep="\t", index=False)

    fig, ax = plt.subplots()
    plot_cluster_identity_range(str(tmp_path), ax)
    plt.close(fig)
