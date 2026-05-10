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
        "ffindex_build_path": "",
        "cstranslate_path": "",
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
        # threads: CPU threads passed to each MMseqs2 / tool invocation.
        # Keep this ≤ the physical cores allocated to a single job/task.
        "threads": 8,
        "tmpdir": "tmp/mmseqs",
        # min_protein_length: proteins shorter than this (amino acids) are
        # removed from the input FASTA before clustering.  Set to 0 to
        # disable filtering.  Dropped proteins are recorded in
        # filtered_short_proteins.tsv for auditing.
        "min_protein_length": 0,
        # min_cluster_size_for_profile: clusters with fewer members than this
        # threshold are skipped during profile building (Step 2).  Singletons
        # (n=1) are still embedded and can join families via KNN.
        # Set to 1 (default) to build profiles for all clusters including
        # singletons; set to 2 to skip singleton profile building.
        "min_cluster_size_for_profile": 1,
    },
    "profiles": {
        "max_members_per_subfamily": 256,
        # parallel_workers: number of subfamily profiles built concurrently.
        # Distinct from mmseqs.threads (which controls per-tool parallelism).
        # On HPC set this to the number of CPU cores in your allocation.
        "parallel_workers": 8,
    },
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
        # parallel_workers: number of concurrent hhalign/hhsearch jobs.
        # Distinct from mmseqs.threads.  Set to core count on HPC.
        "parallel_workers": 8,
    },
    "embed": {
        "esm_model_name": "esm2_t33_650M_UR50D",
        "batch_size": 4,
        "long_seq_policy": "truncate",
        "max_len": 2048,
        "pooling": "mean",
        "device": "cpu",
        # checkpoint_dir: directory to write per-batch embedding checkpoints.
        # If set (and non-empty), the embedding step can resume after a crash
        # or OOM by reloading already-computed batches from disk.
        # Leave empty ("") to disable checkpointing.
        "checkpoint_dir": "",
    },
    "knn": {
        "mode": "knn",
        "k": 100,
        "min_cosine": 0.35,
        "min_len_ratio": 0.5,
        "max_len_ratio": 2.0,
        "device": "cpu",
        # rKCNN-specific parameters (only used when mode="rkcnn")
        "rkcnn_n_subspaces": 50,
        "rkcnn_subspace_fraction": 0.5,
        "rkcnn_n_neighbors": 5,
        "rkcnn_score_threshold": 0.0,
        "rkcnn_weighting": "separation",
        "rkcnn_cascade_topn": 500,
        "rkcnn_random_state": 42,
    },
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
        # min_pident: minimum percent identity (0–100) to accept an mmseqs hit.
        # Legacy key 'min_prob' is still accepted but emits a deprecation warning.
        "min_pident": 0.0,
        "max_evalue": 1e-5,
        "min_segment_len": 120,
        "max_overlap_aa": 30,
    },
    "outputs": {"write_dense_threshold": 2_000_000, "write_matrix_market": True},
    "orthofinder": {
        # subcluster_mode: MMseqs2 algorithm used within each HOG/OG.
        #   "linclust" – fast linear-time clustering (default, good for most sizes)
        #   "cluster"  – sensitive quadratic clustering; better recall for small HOGs
        #   "auto"     – uses "cluster" for HOGs below auto_linclust_min_size,
        #                 "linclust" for larger HOGs; logs the choice per HOG
        "subcluster_mode": "linclust",
        # auto_linclust_min_size: HOGs at or above this member count switch to
        # linclust when subcluster_mode="auto".
        "auto_linclust_min_size": 1000,
        "subcluster_min_seq_id": 0.4,
        "subcluster_coverage": 0.8,
        "subcluster_cov_mode": 1,
        # subcluster_alignment_mode: MMseqs2 --alignment-mode value.
        #   0 = automatic, 1 = only score, 2 = score + end positions,
        #   3 = score + alignment (default; needed for cluster-reassign).
        "subcluster_alignment_mode": 3,
        # subcluster_cluster_reassign: pass --cluster-reassign to mmseqs cluster.
        # Improves cluster membership by reassigning border members.
        # Only valid for "cluster" mode (silently ignored for linclust).
        "subcluster_cluster_reassign": False,
        "min_og_size_for_subclustering": 2,
        # subcluster_threads: CPU threads per individual MMseqs2 OG invocation.
        "subcluster_threads": 4,
        # parallel_og_workers: how many OGs to subcluster concurrently.
        # Total CPU load ≈ parallel_og_workers × subcluster_threads.
        "parallel_og_workers": 4,
    },
}


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------
_RANGE_CHECKS: list[tuple[str, str, float, float]] = [
    ("mmseqs", "min_seq_id", 0.0, 1.0),
    ("mmseqs", "coverage", 0.0, 1.0),
    ("mmseqs", "evalue", 0.0, 1e6),
    ("mmseqs", "threads", 1, 1024),
    ("mmseqs", "min_protein_length", 0, 1_000_000),
    ("mmseqs", "min_cluster_size_for_profile", 1, 100_000),
    ("profiles", "parallel_workers", 1, 1024),
    ("hmm_hmm", "mincov_core", 0.0, 1.0),
    ("hmm_hmm", "min_prob_core", 0.0, 100.0),
    ("hmm_hmm", "mincov_relaxed", 0.0, 1.0),
    ("hmm_hmm", "min_prob_relaxed", 0.0, 100.0),
    ("hmm_hmm", "parallel_workers", 1, 1024),
    ("embed", "batch_size", 1, 512),
    ("embed", "max_len", 64, 100_000),
    ("knn", "k", 1, 100_000),
    ("knn", "min_cosine", -1.0, 1.0),
    ("knn", "rkcnn_n_subspaces", 1, 10_000),
    ("knn", "rkcnn_subspace_fraction", 0.01, 1.0),
    ("knn", "rkcnn_n_neighbors", 1, 10_000),
    ("knn", "rkcnn_score_threshold", 0.0, 1e6),
    ("knn", "rkcnn_cascade_topn", 0, 1_000_000),
    ("graph", "leiden_resolution_strict", 0.01, 100.0),
    ("graph", "leiden_resolution_functional", 0.01, 100.0),
    ("mapping", "profile_cov_min", 0.0, 1.0),
    ("mapping", "min_pident", 0.0, 100.0),
    ("mapping", "max_overlap_aa", 0, 10_000),
    ("orthofinder", "subcluster_min_seq_id", 0.0, 1.0),
    ("orthofinder", "subcluster_coverage", 0.0, 1.0),
    ("orthofinder", "subcluster_cov_mode", 0, 5),
    ("orthofinder", "subcluster_alignment_mode", 0, 4),
    ("orthofinder", "min_og_size_for_subclustering", 1, 1_000_000),
    ("orthofinder", "auto_linclust_min_size", 1, 100_000_000),
    ("orthofinder", "subcluster_threads", 1, 1024),
    ("orthofinder", "parallel_og_workers", 1, 1024),
]

_VALID_EMBED_DEVICES = {"cpu", "cuda", "cuda:0", "cuda:1", "cuda:2", "cuda:3"}
_VALID_EDGE_POLICIES = {"strict", "union", "gated", "downweight_embeddings"}
_VALID_LONG_SEQ_POLICIES = {"truncate", "skip", "full"}
_VALID_HMM_MODES = {"pairwise", "db-search", "mmseqs-profile"}
_VALID_KNN_MODES = {"knn", "rkcnn"}
_VALID_RKCNN_WEIGHTINGS = {"separation", "uniform"}
_VALID_OF_SUBCLUSTER_MODES = {"linclust", "cluster", "auto"}


def validate_config(cfg: dict[str, Any]) -> list[str]:
    """Return a list of human-readable validation errors (empty = OK).

    Note: ``mapping.min_prob`` → ``mapping.min_pident`` normalization is
    performed by :func:`load_config` before this function is called.  This
    function only validates values that are already in their canonical form.
    """
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

    hmm_mode = cfg.get("hmm_hmm", {}).get("mode", "")
    if hmm_mode and hmm_mode not in _VALID_HMM_MODES:
        errors.append(f"hmm_hmm.mode='{hmm_mode}' not in {_VALID_HMM_MODES}")

    knn_mode = cfg.get("knn", {}).get("mode", "")
    if knn_mode and knn_mode not in _VALID_KNN_MODES:
        errors.append(f"knn.mode='{knn_mode}' not in {_VALID_KNN_MODES}")

    knn_device = str(cfg.get("knn", {}).get("device", "cpu"))
    if knn_device not in _VALID_EMBED_DEVICES and not knn_device.startswith("cuda:"):
        errors.append(f"knn.device='{knn_device}' is not valid; use 'cpu' or 'cuda[:N]'")

    rkcnn_w = cfg.get("knn", {}).get("rkcnn_weighting", "")
    if rkcnn_w and rkcnn_w not in _VALID_RKCNN_WEIGHTINGS:
        errors.append(f"knn.rkcnn_weighting='{rkcnn_w}' not in {_VALID_RKCNN_WEIGHTINGS}")

    of_mode = cfg.get("orthofinder", {}).get("subcluster_mode", "")
    if of_mode and of_mode not in _VALID_OF_SUBCLUSTER_MODES:
        errors.append(f"orthofinder.subcluster_mode='{of_mode}' not in {_VALID_OF_SUBCLUSTER_MODES}")

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
    """Load config from a YAML file, deep-merged over defaults.

    Only ``.yaml`` and ``.yml`` files are accepted.  Passing a ``.json`` file
    raises a ``ValueError`` with a clear message.
    Validates parameter ranges and raises ``ValueError`` on problems.

    Backward compatibility
    ----------------------
    ``mapping.min_prob`` is accepted as a legacy alias for ``mapping.min_pident``.
    A deprecation warning is emitted when it is used.
    """
    import warnings
    if not path:
        return deepcopy(DEFAULT_CONFIG)
    p = Path(path)
    if p.suffix.lower() == ".json":
        raise ValueError(
            f"JSON config files are no longer supported: '{path}'. "
            "Please convert your configuration to YAML (.yaml or .yml)."
        )
    if p.suffix.lower() not in {".yaml", ".yml"}:
        raise ValueError(
            f"Unsupported config file extension '{p.suffix}' for '{path}'. "
            "Only .yaml and .yml files are accepted."
        )
    with p.open() as f:
        user = yaml.safe_load(f) or {}

    # Emit deprecation warning for legacy mapping.min_prob before merging
    user_mapping = user.get("mapping", {})
    if "min_prob" in user_mapping and "min_pident" not in user_mapping:
        warnings.warn(
            "Config key 'mapping.min_prob' is deprecated and will be removed in a "
            "future release.  Rename it to 'mapping.min_pident'.",
            DeprecationWarning,
            stacklevel=2,
        )
        user_mapping["min_pident"] = user_mapping.pop("min_prob")
        user["mapping"] = user_mapping

    cfg = _deep_merge(DEFAULT_CONFIG, user)
    errors = validate_config(cfg)
    if errors:
        raise ValueError("Config validation errors:\n  " + "\n  ".join(errors))
    return cfg

