import logging
from pathlib import Path
from unittest.mock import MagicMock, call, patch

import numpy as np
import pandas as pd
import pytest

from plm_cluster.config import load_config
from plm_cluster.pipeline import _build_hhsuite_db, embed, merge_graph, write_matrices


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
# embed() edge-case tests (no real ESM/torch required — calls are mocked)
# ---------------------------------------------------------------------------

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
