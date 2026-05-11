from __future__ import annotations

import concurrent.futures
import json
import os
import re
import tempfile
import threading
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import polars as pl

from .io_utils import FastaRecord, read_fasta, write_fasta
from .runtime import require_executables, run_cmd


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def _sanitize_id(name: str) -> str:
    """Replace non-alphanumeric/underscore characters with underscores.

    Used to produce safe subfamily/OG identifiers from OrthoFinder filenames
    (e.g. ``N0.HOG0000001`` → ``N0_HOG0000001``).
    """
    return re.sub(r"[^A-Za-z0-9_]", "_", name)


def _parse_hhm_len(path: str | Path) -> int:
    for line in Path(path).read_text(errors="ignore").splitlines():
        if line.startswith("LENG"):
            toks = line.split()
            if len(toks) > 1 and toks[1].isdigit():
                return int(toks[1])
    return 0


def _parse_hhr_metrics(path: str | Path) -> dict[str, float]:
    txt = Path(path).read_text(errors="ignore")
    m = {"prob": 0.0, "evalue": 1.0, "pident": np.nan, "aln_len": 0}
    for line in txt.splitlines():
        if "Probab=" in line and "E-value=" in line:
            p = re.search(r"Probab=([0-9.]+)", line)
            e = re.search(r"E-value=([0-9eE+\-.]+)", line)
            a = re.search(r"Aligned_cols=([0-9]+)", line)
            i = re.search(r"Identities=([0-9.]+)%", line)
            if p:
                m["prob"] = float(p.group(1))
            if e:
                m["evalue"] = float(e.group(1))
            if a:
                m["aln_len"] = int(a.group(1))
            if i:
                m["pident"] = float(i.group(1))
            break
    return m


def _parse_hhr_all_hits(path: str | Path) -> list[dict]:
    """Parse all hits from an hhsearch .hhr file.

    Target IDs are normalised: directory prefixes and ``.hhm`` suffixes are
    stripped so that the returned ``target_id`` matches the subfamily ID used
    elsewhere in the pipeline (e.g. ``subfam_000001``).
    """
    lines = Path(path).read_text(errors="ignore").splitlines()
    hits: list[dict] = []
    i = 0
    while i < len(lines):
        if lines[i].startswith(">") and i > 0 and re.match(r"^No \d+\s*$", lines[i - 1]):
            target_id = lines[i][1:].strip().split()[0]
            # Strip directory prefix (ffindex may store full paths)
            target_id = os.path.basename(target_id)
            if target_id.endswith(".hhm"):
                target_id = target_id[:-4]
            m: dict = {"prob": 0.0, "evalue": 1.0, "pident": np.nan, "aln_len": 0}
            for j in range(i + 1, min(i + 20, len(lines))):
                line = lines[j]
                if "Probab=" in line and "E-value=" in line:
                    p = re.search(r"Probab=([0-9.]+)", line)
                    e = re.search(r"E-value=([0-9eE+\-.]+)", line)
                    a = re.search(r"Aligned_cols=([0-9]+)", line)
                    ii = re.search(r"Identities=([0-9.]+)%", line)
                    if p:
                        m["prob"] = float(p.group(1))
                    if e:
                        m["evalue"] = float(e.group(1))
                    if a:
                        m["aln_len"] = int(a.group(1))
                    if ii:
                        m["pident"] = float(ii.group(1))
                    break
            hits.append({"target_id": target_id, **m})
        i += 1
    return hits


_HMM_EDGE_COLS = [
    "q_subfamily_id", "t_subfamily_id", "prob", "evalue", "bits",
    "qcov", "tcov", "aln_len", "pident", "tool", "run_id",
]

# Polars schema for HMM edge DataFrames. "bits" and "pident" are nullable float
# columns; null values are serialised as "NA" in TSV output for backward compat.
_HMM_EDGE_SCHEMA: dict[str, pl.DataType] = {
    "q_subfamily_id": pl.String,
    "t_subfamily_id": pl.String,
    "prob": pl.Float64,
    "evalue": pl.Float64,
    "bits": pl.Float64,
    "qcov": pl.Float64,
    "tcov": pl.Float64,
    "aln_len": pl.Int64,
    "pident": pl.Float64,
    "tool": pl.String,
    "run_id": pl.String,
}


def _load_hmm_progress(path: Path) -> set[tuple[str, str]]:
    done: set[tuple[str, str]] = set()
    if not path.exists():
        return done
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rec = json.loads(line)
            q = rec.get("q")
            t = rec.get("t")
            if q and t and rec.get("status") == "ok":
                done.add((q, t))
        except Exception:
            continue
    return done


def _append_hmm_progress(path: Path, rec: dict, lock: threading.Lock) -> None:
    line = json.dumps(rec, ensure_ascii=False) + "\n"
    with lock:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()


def _build_hhsuite_db(
    db_dir: Path,
    db_prefix: Path,
    hhm_paths: dict[str, str],
    a3m_paths: dict[str, str],
    ffindex_build_bin: str,
    cstranslate_bin: str,
    logger,
) -> None:
    """Build HH-suite ffindex databases (``_hhm``, ``_a3m``, ``_cs219``) for hhsearch.

    HH-suite 3.x ``hhsearch -d <prefix>`` resolves database files as::

        <prefix>_hhm.ffdata    /  <prefix>_hhm.ffindex
        <prefix>_a3m.ffdata    /  <prefix>_a3m.ffindex
        <prefix>_cs219.ffdata  /  <prefix>_cs219.ffindex

    All three databases must exist for hhsearch to run without error.
    This helper writes the file lists to temporary files (avoiding platform
    issues with ``/dev/stdin``) and calls ``ffindex_build`` for ``_hhm`` and
    ``_a3m``, then ``cstranslate`` to produce the ``_cs219`` database.
    """
    db_dir.mkdir(parents=True, exist_ok=True)
    ffdata_hhm = Path(str(db_prefix) + "_hhm.ffdata")
    ffindex_hhm = Path(str(db_prefix) + "_hhm.ffindex")
    ffdata_a3m = Path(str(db_prefix) + "_a3m.ffdata")
    ffindex_a3m = Path(str(db_prefix) + "_a3m.ffindex")
    ffdata_cs219 = Path(str(db_prefix) + "_cs219.ffdata")
    ffindex_cs219 = Path(str(db_prefix) + "_cs219.ffindex")

    # Build _hhm database
    if not (ffdata_hhm.exists() and ffindex_hhm.exists()):
        logger.info("Building HH-suite _hhm ffindex DB from %d profiles", len(hhm_paths))
        hhm_list = sorted(hhm_paths.values())
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".txt", dir=str(db_dir), delete=False,
        ) as fh:
            fh.write("\n".join(hhm_list) + "\n")
            hhm_list_file = fh.name
        try:
            run_cmd(
                [ffindex_build_bin, "-s", "-f", hhm_list_file,
                 str(ffdata_hhm), str(ffindex_hhm)],
                logger,
            )
        finally:
            Path(hhm_list_file).unlink(missing_ok=True)
    else:
        logger.info("HH-suite _hhm DB already exists at %s, reusing.", str(db_prefix))

    # Build _a3m database
    if not (ffdata_a3m.exists() and ffindex_a3m.exists()):
        valid_a3m = {k: v for k, v in a3m_paths.items() if v and Path(v).exists()}
        if valid_a3m:
            logger.info("Building HH-suite _a3m ffindex DB from %d MSAs", len(valid_a3m))
            a3m_list = sorted(valid_a3m.values())
            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".txt", dir=str(db_dir), delete=False,
            ) as fh:
                fh.write("\n".join(a3m_list) + "\n")
                a3m_list_file = fh.name
            try:
                run_cmd(
                    [ffindex_build_bin, "-s", "-f", a3m_list_file,
                     str(ffdata_a3m), str(ffindex_a3m)],
                    logger,
                )
            finally:
                Path(a3m_list_file).unlink(missing_ok=True)
        else:
            logger.info("No .a3m files available; _a3m database will be empty.")
            ffdata_a3m.write_bytes(b"")
            ffindex_a3m.write_text("")
    else:
        logger.info("HH-suite _a3m DB already exists at %s, reusing.", str(db_prefix))

    # Build _cs219 database from the _a3m database using cstranslate
    if not (ffdata_cs219.exists() and ffindex_cs219.exists()):
        logger.info("Building HH-suite _cs219 ffindex DB from _a3m DB at %s", str(db_prefix))
        run_cmd(
            [cstranslate_bin,
             "-i", str(db_prefix) + "_a3m",
             "-o", str(db_prefix) + "_cs219",
             "-x", "0.3", "-c", "4",
             "-I", "a3m",
             "-b"],
            logger,
        )
    else:
        logger.info("HH-suite _cs219 DB already exists at %s, reusing.", str(db_prefix))


def _filter_raw_to_core_relaxed(raw: pl.DataFrame, hc: dict, out: Path) -> None:
    """Compute and write core and relaxed edge TSVs from a raw edge DataFrame."""
    raw2 = raw.with_columns(
        pl.min_horizontal(pl.col("qcov"), pl.col("tcov")).alias("mincov"),
    ).with_columns(
        (pl.col("prob") / 100.0 * pl.col("mincov")).alias("edge_weight"),
    )

    out_cols = ["q_subfamily_id", "t_subfamily_id", "edge_weight", "prob", "evalue",
                "qcov", "tcov", "mincov", "aln_len", "source"]

    core = raw2.filter(
        (pl.col("mincov") >= float(hc["mincov_core"]))
        & ((pl.col("prob") >= float(hc["min_prob_core"])) | (pl.col("evalue") <= float(hc["max_evalue_core"])))
        & (pl.col("aln_len") >= int(hc["min_aln_len_core"]))
    ).with_columns(pl.lit("hmm_hmm_core").alias("source")).select(out_cols)
    core.write_csv(out / "hmm_hmm_edges_core.tsv", separator="\t")

    relaxed = raw2.filter(
        (pl.col("mincov") >= float(hc["mincov_relaxed"]))
        & ((pl.col("prob") >= float(hc["min_prob_relaxed"])) | (pl.col("evalue") <= float(hc["max_evalue_relaxed"])))
        & (pl.col("aln_len") >= int(hc["min_aln_len_relaxed"]))
    ).with_columns(pl.lit("hmm_hmm_relaxed").alias("source")).select(out_cols)
    relaxed.write_csv(out / "hmm_hmm_edges_relaxed.tsv", separator="\t")


def merge_hmm_shards(outdir: str, config: dict, logger, resume: bool = False) -> str:
    """Merge per-shard raw TSV files into a combined raw TSV and produce core/relaxed TSVs."""
    out = Path(outdir)
    merged_path = out / "hmm_hmm_edges_raw.tsv"
    if resume and merged_path.exists():
        logger.info("Resume: merged HMM-HMM edges already exist at %s, skipping.", str(merged_path))
        return str(merged_path)
    shard_files = sorted(out.glob("hmm_hmm_edges_raw.shard_*.tsv"))
    if not shard_files:
        raise FileNotFoundError(f"No shard TSV files found in {outdir}")
    logger.info("Merging %d shard TSV(s) from %s", len(shard_files), outdir)
    dfs = [pl.read_csv(f, separator="\t", null_values=["NA"], schema_overrides=_HMM_EDGE_SCHEMA) for f in shard_files]
    raw = pl.concat(dfs).sort(["q_subfamily_id", "t_subfamily_id"])
    raw.write_csv(merged_path, separator="\t", null_value="NA")
    _filter_raw_to_core_relaxed(raw, config["hmm_hmm"], out)
    logger.info("Merged raw TSV written to %s", str(merged_path))
    return str(merged_path)


def mmseqs_cluster(proteins_fasta: str, outdir: str, config: dict, logger, resume: bool = False) -> dict[str, str]:
    """Step 1: Cluster proteins into subfamilies using MMseqs2.

    Pre-filtering
    -------------
    When ``mmseqs.min_protein_length > 0``, proteins shorter than that threshold
    are removed before clustering.  Dropped proteins are written to
    ``filtered_short_proteins.tsv`` (columns: ``protein_id``, ``length_aa``).

    Extended stats
    --------------
    ``subfamily_stats.tsv`` now includes the following extra columns:

    * ``is_singleton`` – 1 when the cluster has exactly one member, else 0.
    * ``min_length_aa``, ``max_length_aa``, ``mean_length_aa``, ``std_length_aa``
      – length distribution of member sequences.
    * ``min_pident``, ``max_pident``, ``mean_pident`` – identity range across
      within-cluster pairwise alignments produced by ``mmseqs align``.
      Null for singletons.

    Summary report
    --------------
    ``mmseqs_cluster_report.tsv`` (one wide row) summarises counts of singletons,
    multi-member clusters, filtered proteins, and profile-eligible clusters.
    """
    out = _ensure_dir(outdir)
    if resume and (out / "subfamily_map.tsv").exists():
        logger.info("Resume: mmseqs-cluster outputs already exist in %s, skipping.", str(out))
        return require_executables(["mmseqs"], config["tools"])
    mm = config["mmseqs"]
    tools = require_executables(["mmseqs"], config["tools"])
    tmpdir = _ensure_dir(mm["tmpdir"])
    min_len = int(mm.get("min_protein_length", 0))
    min_cluster_size_for_profile = int(mm.get("min_cluster_size_for_profile", 1))

    # ------------------------------------------------------------------
    # Load sequences and apply length pre-filter
    # ------------------------------------------------------------------
    all_records = list(read_fasta(proteins_fasta))
    n_input = len(all_records)
    if min_len > 0:
        kept_records = [r for r in all_records if len(r.seq) >= min_len]
        filtered_records = [r for r in all_records if len(r.seq) < min_len]
        if filtered_records:
            pl.DataFrame({
                "protein_id": [r.id for r in filtered_records],
                "length_aa": [len(r.seq) for r in filtered_records],
            }).write_csv(out / "filtered_short_proteins.tsv", separator="\t")
            logger.info(
                "Length pre-filter (min_protein_length=%d): removed %d / %d proteins",
                min_len, len(filtered_records), n_input,
            )
    else:
        kept_records = all_records
        filtered_records = []

    n_filtered = len(filtered_records)
    n_clustered = len(kept_records)

    # Write the (possibly filtered) FASTA for MMseqs2
    filtered_fasta = out / "proteins_for_clustering.faa"
    write_fasta(filtered_fasta, kept_records)
    seqs = {r.id: r.seq for r in kept_records}

    # ------------------------------------------------------------------
    # Run MMseqs2 clustering
    # ------------------------------------------------------------------
    db = out / "proteinsDB"
    clu = out / "clusterDB"
    tsv = out / "clusters.tsv"
    rep_db = out / "repDB"
    run_cmd([tools["mmseqs"], "createdb", str(filtered_fasta), str(db)], logger)
    mmseqs_cmd = [
        tools["mmseqs"], mm["mode"], str(db), str(clu), str(tmpdir),
        "--min-seq-id", str(mm["min_seq_id"]),
        "-c", str(mm["coverage"]),
        "--cov-mode", str(mm["cov_mode"]),
        "-e", str(mm["evalue"]),
        "--threads", str(mm["threads"]),
    ]
    if str(mm["mode"]).lower() != "linclust":
        mmseqs_cmd.extend(["-s", str(mm["sensitivity"])])
    run_cmd(mmseqs_cmd, logger)
    run_cmd([tools["mmseqs"], "createtsv", str(db), str(db), str(clu), str(tsv)], logger)
    run_cmd([tools["mmseqs"], "result2repseq", str(db), str(clu), str(rep_db)], logger)
    run_cmd([tools["mmseqs"], "result2flat", str(db), str(db), str(rep_db), str(out / "subfamily_reps_raw.faa")], logger)

    raw = pl.read_csv(tsv, separator="\t", has_header=False, new_columns=["rep_protein_id", "protein_id"])
    reps_sorted = sorted(raw["rep_protein_id"].unique().to_list())
    rep_to_sub = {r: f"subfam_{i:06d}" for i, r in enumerate(reps_sorted)}

    sub_lookup = pl.DataFrame({
        "rep_protein_id": list(rep_to_sub.keys()),
        "subfamily_id": list(rep_to_sub.values()),
    })
    raw = (
        raw.join(sub_lookup, on="rep_protein_id", how="left")
        .with_columns((pl.col("rep_protein_id") == pl.col("protein_id")).cast(pl.Int8).alias("is_rep"))
    )
    mapping = raw.select(["protein_id", "subfamily_id", "is_rep"]).sort(["subfamily_id", "protein_id"])
    mapping.write_csv(out / "subfamily_map.tsv", separator="\t")

    rep_records = [FastaRecord(rep_to_sub[rid], seqs[rid]) for rid in reps_sorted if rid in seqs]
    write_fasta(out / "subfamily_reps.faa", rep_records)

    # ------------------------------------------------------------------
    # Compute within-cluster identity stats via mmseqs align
    # ------------------------------------------------------------------
    align_db = out / "alignDB"
    align_tsv = out / "align.tsv"
    try:
        run_cmd(
            [tools["mmseqs"], "align", str(db), str(db), str(clu), str(align_db),
             "--threads", str(mm["threads"]), "-a"],
            logger,
        )
        run_cmd(
            [tools["mmseqs"], "convertalis", str(db), str(db), str(align_db), str(align_tsv),
             "--format-output", "query,target,pident",
             "--threads", str(mm["threads"])],
            logger,
        )
    except Exception as exc:
        logger.warning("mmseqs align for identity stats failed (skipping): %s", exc)
        align_tsv = None

    # Build per-cluster pident stats from the alignment TSV
    cluster_pident: dict[str, dict[str, float]] = {}
    if align_tsv is not None and Path(str(align_tsv)).exists() and Path(str(align_tsv)).stat().st_size > 0:
        try:
            aln_df = pl.read_csv(align_tsv, separator="\t", has_header=False,
                                 new_columns=["query", "target", "pident"])
            # Map query protein → subfamily
            pid_to_sub = dict(zip(raw["protein_id"].to_list(), raw["subfamily_id"].to_list()))
            aln_df = aln_df.filter(pl.col("query") != pl.col("target"))

            def _map_pid_to_subfam(p: str) -> str:
                return pid_to_sub.get(p, "")

            aln_df = aln_df.with_columns(
                pl.col("query").map_elements(_map_pid_to_subfam, return_dtype=pl.String).alias("subfamily_id")
            ).filter(pl.col("subfamily_id") != "")
            pident_stats = (
                aln_df.group_by("subfamily_id")
                .agg([
                    pl.col("pident").min().alias("min_pident"),
                    pl.col("pident").max().alias("max_pident"),
                    pl.col("pident").mean().alias("mean_pident"),
                ])
            )
            for row in pident_stats.iter_rows(named=True):
                cluster_pident[row["subfamily_id"]] = {
                    "min_pident": row["min_pident"],
                    "max_pident": row["max_pident"],
                    "mean_pident": row["mean_pident"],
                }
        except Exception as exc:
            logger.warning("Failed to parse alignment TSV for identity stats: %s", exc)

    # ------------------------------------------------------------------
    # Build extended subfamily_stats.tsv
    # ------------------------------------------------------------------
    stats = raw.group_by("subfamily_id").agg(pl.col("protein_id").len().alias("n_members"))
    reps_df = (
        raw.filter(pl.col("is_rep") == 1)
        .select(["subfamily_id", "protein_id"])
        .rename({"protein_id": "rep_protein_id"})
    )
    stats = stats.join(reps_df, on="subfamily_id", how="left")

    # Collect all member lengths per cluster
    subfam_member_ids: dict[str, list[str]] = {}
    for row in raw.iter_rows(named=True):
        subfam_member_ids.setdefault(row["subfamily_id"], []).append(row["protein_id"])

    subfam_ids_list = stats["subfamily_id"].to_list()
    rep_ids = stats["rep_protein_id"].to_list()

    rep_lengths = [len(seqs.get(rid, "")) if rid else 0 for rid in rep_ids]
    min_lens, max_lens, mean_lens, std_lens = [], [], [], []
    is_singleton_col = []
    min_pident_col, max_pident_col, mean_pident_col = [], [], []

    for sfid in subfam_ids_list:
        member_lens = [len(seqs.get(pid, "")) for pid in subfam_member_ids.get(sfid, [])]
        if member_lens:
            min_lens.append(min(member_lens))
            max_lens.append(max(member_lens))
            mean_lens.append(sum(member_lens) / len(member_lens))
            n = len(member_lens)
            if n > 1:
                variance = sum((x - mean_lens[-1]) ** 2 for x in member_lens) / n
                std_lens.append(variance ** 0.5)
            else:
                std_lens.append(0.0)
        else:
            min_lens.append(0)
            max_lens.append(0)
            mean_lens.append(0.0)
            std_lens.append(0.0)

        n_mem = len(subfam_member_ids.get(sfid, []))
        is_singleton_col.append(1 if n_mem == 1 else 0)

        pident_info = cluster_pident.get(sfid, {})
        min_pident_col.append(pident_info.get("min_pident", None))
        max_pident_col.append(pident_info.get("max_pident", None))
        mean_pident_col.append(pident_info.get("mean_pident", None))

    stats = stats.with_columns([
        pl.Series("rep_length_aa", rep_lengths),
        pl.Series("is_singleton", is_singleton_col, dtype=pl.Int8),
        pl.Series("min_length_aa", min_lens),
        pl.Series("max_length_aa", max_lens),
        pl.Series("mean_length_aa", mean_lens),
        pl.Series("std_length_aa", std_lens),
        pl.Series("min_pident", min_pident_col, dtype=pl.Float64),
        pl.Series("max_pident", max_pident_col, dtype=pl.Float64),
        pl.Series("mean_pident", mean_pident_col, dtype=pl.Float64),
    ])
    stats.sort("subfamily_id").write_csv(out / "subfamily_stats.tsv", separator="\t")

    # ------------------------------------------------------------------
    # Write summary report
    # ------------------------------------------------------------------
    n_clusters_total = len(reps_sorted)
    n_singletons = int(stats["is_singleton"].sum())
    n_clusters_2plus = n_clusters_total - n_singletons
    n_clusters_above_profile_threshold = int(
        (stats["n_members"] >= min_cluster_size_for_profile).sum()
    )
    report = pl.DataFrame({
        "n_proteins_input": [n_input],
        "n_proteins_filtered_short": [n_filtered],
        "n_proteins_clustered": [n_clustered],
        "n_clusters_total": [n_clusters_total],
        "n_singletons": [n_singletons],
        "n_clusters_2plus": [n_clusters_2plus],
        "n_clusters_above_min_profile_size": [n_clusters_above_profile_threshold],
    })
    report.write_csv(out / "mmseqs_cluster_report.tsv", separator="\t")

    median_size = float(stats["n_members"].median())
    logger.info(
        "MMseqs2 clustering complete: %d proteins -> %d subfamilies "
        "(median size: %.0f, singletons: %d, clusters ≥2: %d, "
        "eligible for profiles [≥%d]: %d)",
        len(raw), n_clusters_total, median_size,
        n_singletons, n_clusters_2plus,
        min_cluster_size_for_profile, n_clusters_above_profile_threshold,
    )

    # Generate publication-quality cluster QC plots directly into this step's output folder.
    from .qc_plots import generate_mmseqs_cluster_plots  # noqa: PLC0415
    generate_mmseqs_cluster_plots(out, logger=logger, resume=resume)

    return tools


def _og_checkpoint_path(checkpoint_dir: Path, og_stem: str) -> Path:
    """Return the path to the per-OG checkpoint JSON file."""
    return checkpoint_dir / f"{og_stem}.json"


def _load_og_checkpoint(
    checkpoint_dir: Path, og_stem: str
) -> "tuple[list[dict], list[dict], list[FastaRecord], list[FastaRecord], int] | None":
    """Load per-OG results from a checkpoint file.

    Returns ``None`` if no checkpoint exists for *og_stem*.
    """
    cp = _og_checkpoint_path(checkpoint_dir, og_stem)
    if not cp.exists():
        return None
    with cp.open() as fh:
        data = json.load(fh)
    reps = [FastaRecord(r["id"], r["seq"]) for r in data["reps"]]
    combined = [FastaRecord(r["id"], r["seq"]) for r in data["combined"]]
    return data["rows"], data["og_rows"], reps, combined, data["n_singletons"]


def _save_og_checkpoint(
    checkpoint_dir: Path,
    og_stem: str,
    rows: list[dict],
    og_rows: list[dict],
    reps: "list[FastaRecord]",
    combined: "list[FastaRecord]",
    n_singletons: int,
) -> None:
    """Atomically write per-OG results to a checkpoint JSON file."""
    data = {
        "rows": rows,
        "og_rows": og_rows,
        "reps": [{"id": r.id, "seq": r.seq} for r in reps],
        "combined": [{"id": r.id, "seq": r.seq} for r in combined],
        "n_singletons": n_singletons,
    }
    cp = _og_checkpoint_path(checkpoint_dir, og_stem)
    tmp = cp.with_suffix(".tmp")
    with tmp.open("w") as fh:
        json.dump(data, fh)
    tmp.rename(cp)


def orthofinder_cluster(og_dir: str, outdir: str, config: dict, logger, resume: bool = False) -> dict[str, str]:
    """Step 1 (OrthoFinder mode): Subcluster each HOG/OG FASTA into subfamilies using MMseqs2.

    Scans *og_dir* for ``*.faa`` and ``*.fa`` files produced by OrthoFinder
    (HOGs or OGs).  Each file is subclustered independently so that subfamily
    IDs are scoped to their parent orthogroup, preventing cross-OG collisions.
    Singletons (OGs below *min_og_size_for_subclustering*) are passed through
    directly without invoking MMseqs2.

    Produces the same output schema as :func:`mmseqs_cluster` so that all
    downstream steps (build_profiles, embed, …) work without modification.
    Two additional files are written:

    * ``proteins_combined.faa`` — all proteins from all OGs concatenated;
      required by ``build_profiles`` and ``map_proteins_to_families``.
    * ``og_subfamily_map.tsv``  — provenance table mapping each subfamily back
      to its source OG identifier.

    A ``og_checkpoints/`` subdirectory is written inside *outdir* during
    processing.  Each successfully processed OG is saved as
    ``og_checkpoints/<og_stem>.json``.  When *resume* is ``True`` and
    ``subfamily_map.tsv`` does not yet exist (i.e. the previous run did not
    finish), OGs that already have a checkpoint are loaded from disk and
    skipped, while OGs without a checkpoint (including any that timed out
    mid-run) are (re-)processed from scratch.  This allows a long run to be
    safely interrupted and resumed without using any partial per-OG data.
    """
    out = _ensure_dir(outdir)
    if resume and (out / "subfamily_map.tsv").exists():
        logger.info("Resume: orthofinder-cluster outputs already exist in %s, skipping.", str(out))
        return require_executables(["mmseqs"], config["tools"])

    tools = require_executables(["mmseqs"], config["tools"])
    of_cfg = config.get("orthofinder", {})
    min_seq_id = float(of_cfg.get("subcluster_min_seq_id", 0.4))
    coverage = float(of_cfg.get("subcluster_coverage", 0.8))
    cov_mode = int(of_cfg.get("subcluster_cov_mode", 1))
    alignment_mode = int(of_cfg.get("subcluster_alignment_mode", 3))
    cluster_reassign = bool(of_cfg.get("subcluster_cluster_reassign", False))
    subcluster_mode = str(of_cfg.get("subcluster_mode", "linclust")).lower()
    auto_linclust_min_size = int(of_cfg.get("auto_linclust_min_size", 1000))
    min_og_size = int(of_cfg.get("min_og_size_for_subclustering", 2))
    min_protein_length = int(config.get("mmseqs", {}).get("min_protein_length", 0))
    # per-OG MMseqs2 thread count (subcluster_threads) and number of OGs
    # processed concurrently (parallel_og_workers) are kept separate so that
    # total CPU load = parallel_og_workers × subcluster_threads.
    og_threads = str(int(of_cfg.get("subcluster_threads",
                         config.get("mmseqs", {}).get("threads", 4))))
    parallel_ogs = int(of_cfg.get("parallel_og_workers", 4))
    tmpdir = _ensure_dir(config.get("mmseqs", {}).get("tmpdir", "tmp/mmseqs"))

    og_dir_path = Path(og_dir)
    og_files = sorted(og_dir_path.glob("*.faa")) + sorted(og_dir_path.glob("*.fa"))
    # Deduplicate in case a path matches both patterns (e.g. file named x.fa.faa)
    seen: set[Path] = set()
    unique_og_files: list[Path] = []
    for f in og_files:
        if f not in seen:
            seen.add(f)
            unique_og_files.append(f)
    og_files = unique_og_files

    if not og_files:
        raise FileNotFoundError(
            f"No *.faa or *.fa files found in '{og_dir}'. "
            "Please point --og_dir at a directory of OrthoFinder HOG or OG FASTA files."
        )

    # Per-HOG checkpoint directory: stores one JSON per successfully processed OG so
    # that a resumable run can skip already-completed OGs without re-running them.
    checkpoint_dir = _ensure_dir(out / "og_checkpoints")

    # Count how many OGs already have checkpoints (for the progress log message).
    n_already_done = sum(
        1 for f in og_files
        if _og_checkpoint_path(checkpoint_dir, f.stem).exists()
    ) if resume else 0

    logger.info(
        "OrthoFinder subclustering: found %d OG/HOG FASTA files in %s "
        "(mode=%s, parallel_og_workers=%d, subcluster_threads=%s%s)",
        len(og_files), og_dir, subcluster_mode, parallel_ogs, og_threads,
        f", resuming — {n_already_done}/{len(og_files)} OGs already checkpointed" if resume else "",
    )

    # ------------------------------------------------------------------
    # Per-OG worker function (may run in a thread pool)
    # ------------------------------------------------------------------
    def _process_og(og_file: Path) -> tuple[list[dict], list[dict], list[FastaRecord], list[FastaRecord], int]:
        """Process a single OG FASTA → (all_rows, og_subfam_rows, rep_records, combined_seqs, n_singletons)."""
        og_stem = og_file.stem
        og_id = _sanitize_id(og_stem)

        records = read_fasta(str(og_file))
        if not records:
            logger.warning("OG file %s is empty, skipping.", str(og_file))
            return [], [], [], [], 0

        # Apply protein length pre-filter
        if min_protein_length > 0:
            filtered_out = [r for r in records if len(r.seq) < min_protein_length]
            records = [r for r in records if len(r.seq) >= min_protein_length]
            if filtered_out:
                logger.info(
                    "OG %s: removed %d short protein(s) (min_protein_length=%d)",
                    og_stem, len(filtered_out), min_protein_length,
                )
            if not records:
                logger.warning("OG file %s: all proteins removed by length filter, skipping.", str(og_file))
                return [], [], [], [], 0

        seqs = {r.id: r.seq for r in records}
        sorted_protein_ids = sorted(seqs.keys())
        combined = list(records)

        if len(records) < min_og_size:
            rows: list[dict] = []
            og_rows: list[dict] = []
            reps: list[FastaRecord] = []
            for i, pid in enumerate(sorted_protein_ids):
                subfam_id = f"{og_id}_subfam_{i:06d}"
                rows.append({"protein_id": pid, "subfamily_id": subfam_id, "is_rep": 1})
                og_rows.append({"subfamily_id": subfam_id, "og_id": og_stem})
                reps.append(FastaRecord(subfam_id, seqs[pid]))
            return rows, og_rows, reps, combined, len(records)

        # Determine effective clustering algorithm for this OG
        if subcluster_mode == "auto":
            effective_algo = "linclust" if len(records) >= auto_linclust_min_size else "cluster"
            logger.info("OG %s: %d members → using mmseqs %s (auto_linclust_min_size=%d)",
                        og_stem, len(records), effective_algo, auto_linclust_min_size)
        else:
            effective_algo = subcluster_mode  # "linclust" or "cluster"

        with tempfile.TemporaryDirectory(dir=str(tmpdir)) as og_tmp:
            og_tmp_path = Path(og_tmp)
            fa_path = og_tmp_path / "og.faa"
            write_fasta(str(fa_path), records)

            db = og_tmp_path / "proteinsDB"
            clu = og_tmp_path / "clusterDB"
            tsv = og_tmp_path / "clusters.tsv"
            rep_db = og_tmp_path / "repDB"
            rep_faa = og_tmp_path / "reps.faa"

            run_cmd([tools["mmseqs"], "createdb", str(fa_path), str(db)], logger)

            cluster_cmd = [
                tools["mmseqs"], effective_algo, str(db), str(clu),
                str(og_tmp_path / "mmseqs_tmp"),
                "--min-seq-id", str(min_seq_id),
                "-c", str(coverage),
                "--cov-mode", str(cov_mode),
                "--threads", og_threads,
            ]
            if effective_algo == "cluster":
                cluster_cmd += ["--alignment-mode", str(alignment_mode)]
                if cluster_reassign:
                    cluster_cmd += ["--cluster-reassign"]
            run_cmd(cluster_cmd, logger)

            run_cmd([tools["mmseqs"], "createtsv", str(db), str(db), str(clu), str(tsv)], logger)
            run_cmd([tools["mmseqs"], "result2repseq", str(db), str(clu), str(rep_db)], logger)
            run_cmd([tools["mmseqs"], "result2flat", str(db), str(db), str(rep_db), str(rep_faa)], logger)

            raw = pl.read_csv(str(tsv), separator="\t", has_header=False,
                              new_columns=["rep_protein_id", "protein_id"])
            reps_sorted = sorted(raw["rep_protein_id"].unique().to_list())
            rep_to_sub = {r: f"{og_id}_subfam_{i:06d}" for i, r in enumerate(reps_sorted)}

            rep_seqs = {r.id: r.seq for r in read_fasta(str(rep_faa))}

            og_rows_inner: list[dict] = []
            reps_inner: list[FastaRecord] = []
            for rep_pid, subfam_id in rep_to_sub.items():
                og_rows_inner.append({"subfamily_id": subfam_id, "og_id": og_stem})
                seq = seqs[rep_pid] if rep_pid in seqs else rep_seqs.get(rep_pid, "")
                reps_inner.append(FastaRecord(subfam_id, seq))

            rows_inner: list[dict] = []
            for row in raw.iter_rows(named=True):
                subfam_id = rep_to_sub[row["rep_protein_id"]]
                is_rep = 1 if row["rep_protein_id"] == row["protein_id"] else 0
                rows_inner.append({
                    "protein_id": row["protein_id"],
                    "subfamily_id": subfam_id,
                    "is_rep": is_rep,
                })

        return rows_inner, og_rows_inner, reps_inner, combined, 0

    # ------------------------------------------------------------------
    # Per-HOG checkpoint wrapper: on resume, loads existing checkpoint;
    # otherwise processes and saves a new checkpoint.
    # ------------------------------------------------------------------
    def _run_or_load_og(og_file: Path) -> tuple[list[dict], list[dict], list[FastaRecord], list[FastaRecord], int]:
        og_stem = og_file.stem
        if resume:
            cached = _load_og_checkpoint(checkpoint_dir, og_stem)
            if cached is not None:
                return cached
        result = _process_og(og_file)
        _save_og_checkpoint(checkpoint_dir, og_stem, *result)
        return result

    # ------------------------------------------------------------------
    # Process all OGs (optionally in parallel)
    # ------------------------------------------------------------------
    all_rows: list[dict] = []
    og_subfamily_rows: list[dict] = []
    rep_records: list[FastaRecord] = []
    combined_seqs: list[FastaRecord] = []
    n_singletons = 0

    if parallel_ogs > 1 and len(og_files) > 1:
        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_ogs) as pool:
            future_map = {pool.submit(_run_or_load_og, f): f for f in og_files}
            for fut in concurrent.futures.as_completed(future_map):
                og_file = future_map[fut]
                try:
                    rows, og_rows, reps, combined, n_sing = fut.result()
                except Exception as exc:
                    logger.error("OG processing FAILED for %s: %s", str(og_file), exc)
                    continue
                all_rows.extend(rows)
                og_subfamily_rows.extend(og_rows)
                rep_records.extend(reps)
                combined_seqs.extend(combined)
                n_singletons += n_sing
    else:
        for og_file in og_files:
            rows, og_rows, reps, combined, n_sing = _run_or_load_og(og_file)
            all_rows.extend(rows)
            og_subfamily_rows.extend(og_rows)
            rep_records.extend(reps)
            combined_seqs.extend(combined)
            n_singletons += n_sing

    if not all_rows:
        raise RuntimeError("orthofinder_cluster: no proteins were processed. Check input FASTA files.")

    # Write combined proteins FASTA (needed by build_profiles and map_proteins_to_families)
    write_fasta(str(out / "proteins_combined.faa"), combined_seqs)

    # Write OG provenance table
    pl.DataFrame(og_subfamily_rows).sort(["og_id", "subfamily_id"]).write_csv(
        str(out / "og_subfamily_map.tsv"), separator="\t"
    )

    # Write subfamily_map.tsv (same schema as mmseqs_cluster output)
    mapping_df = pl.DataFrame(all_rows).sort(["subfamily_id", "protein_id"])
    mapping_df.write_csv(str(out / "subfamily_map.tsv"), separator="\t")

    # Write subfamily_reps.faa (sorted by subfamily_id for determinism)
    rep_records_sorted = sorted(rep_records, key=lambda r: r.id)
    write_fasta(str(out / "subfamily_reps.faa"), rep_records_sorted)

    # Write extended subfamily_stats.tsv (matches mmseqs_cluster output schema)
    stats = (
        mapping_df
        .group_by("subfamily_id")
        .agg(pl.col("protein_id").len().alias("n_members"))
    )
    reps_df = (
        mapping_df.filter(pl.col("is_rep") == 1)
        .select(["subfamily_id", "protein_id"])
        .rename({"protein_id": "rep_protein_id"})
    )
    stats = stats.join(reps_df, on="subfamily_id", how="left")
    all_seqs = {r.id: r.seq for r in combined_seqs}
    rep_ids_list = stats["rep_protein_id"].to_list()

    # Build per-subfamily member sequences for length stats
    subfam_member_ids_of: dict[str, list[str]] = {}
    for row in mapping_df.iter_rows(named=True):
        subfam_member_ids_of.setdefault(row["subfamily_id"], []).append(row["protein_id"])

    of_subfam_ids = stats["subfamily_id"].to_list()
    rep_lengths = [len(all_seqs.get(rid, "")) if rid else 0 for rid in rep_ids_list]
    is_singleton_col: list[int] = []
    min_lens: list[int] = []
    max_lens: list[int] = []
    mean_lens: list[float] = []
    std_lens: list[float] = []

    for sfid in of_subfam_ids:
        member_lens = [len(all_seqs.get(pid, "")) for pid in subfam_member_ids_of.get(sfid, [])]
        n_mem = len(member_lens)
        is_singleton_col.append(1 if n_mem == 1 else 0)
        if member_lens:
            mn = min(member_lens)
            mx = max(member_lens)
            avg = sum(member_lens) / n_mem
            if n_mem > 1:
                variance = sum((x - avg) ** 2 for x in member_lens) / n_mem
                std = variance ** 0.5
            else:
                std = 0.0
        else:
            mn, mx, avg, std = 0, 0, 0.0, 0.0
        min_lens.append(mn)
        max_lens.append(mx)
        mean_lens.append(avg)
        std_lens.append(std)

    stats = stats.with_columns([
        pl.Series("rep_length_aa", rep_lengths),
        pl.Series("is_singleton", is_singleton_col, dtype=pl.Int8),
        pl.Series("min_length_aa", min_lens),
        pl.Series("max_length_aa", max_lens),
        pl.Series("mean_length_aa", mean_lens),
        pl.Series("std_length_aa", std_lens),
        # pident stats not computed for OrthoFinder mode (no within-cluster align step)
        pl.Series("min_pident", [None] * len(of_subfam_ids), dtype=pl.Float64),
        pl.Series("max_pident", [None] * len(of_subfam_ids), dtype=pl.Float64),
        pl.Series("mean_pident", [None] * len(of_subfam_ids), dtype=pl.Float64),
    ])
    stats.sort("subfamily_id").write_csv(str(out / "subfamily_stats.tsv"), separator="\t")

    n_subfamilies = len(rep_records_sorted)
    n_proteins = len(all_rows)
    n_ogs = len(og_files)
    n_of_singletons = int(stats["is_singleton"].sum()) if len(stats) > 0 else 0
    pct_singletons = 100.0 * n_singletons / n_proteins if n_proteins else 0.0
    median_size = float(stats["n_members"].median()) if len(stats) > 0 else 0.0
    logger.info(
        "OrthoFinder subclustering complete: %d OGs → %d subfamilies from %d proteins "
        "(median size: %.0f, singletons: %d [%.1f%%])",
        n_ogs, n_subfamilies, n_proteins, median_size, n_of_singletons, pct_singletons,
    )
    return tools


def build_profiles(
    proteins_fasta: str, subfamily_map: str, outdir: str, config: dict, logger, resume: bool = False
) -> dict[str, str]:
    """Step 2: Build MSAs (MAFFT) and profile HMMs (hhmake) for each subfamily.

    Subfamilies with fewer members than ``mmseqs.min_cluster_size_for_profile``
    are skipped — no profile is built for them.  Skipped subfamilies are still
    included in ``subfamily_profile_index.tsv`` with ``profile_built=False`` and
    an appropriate ``skipped_reason`` so that downstream steps can identify them.

    Singletons that are skipped still appear in ``subfamily_reps.faa`` and are
    embedded in Step 3, so they can join families via KNN edges in Step 4.
    """
    out = _ensure_dir(outdir)
    tools = require_executables(["mafft", "hhmake"], config["tools"])
    if not Path(subfamily_map).exists():
        raise RuntimeError(
            f"subfamily_map not found: {subfamily_map}. Run mmseqs-cluster first."
        )
    smap = pl.read_csv(subfamily_map, separator="\t")
    seqs = {r.id: r.seq for r in read_fasta(proteins_fasta)}
    cap = int(config["profiles"]["max_members_per_subfamily"])
    min_cluster_size_for_profile = int(config.get("mmseqs", {}).get("min_cluster_size_for_profile", 1))

    def _process_subfam(subfam, members):
        fa = out / f"{subfam}.faa"
        a3m = out / f"{subfam}.a3m"
        hhm = out / f"{subfam}.hhm"
        write_fasta(fa, [FastaRecord(pid, seqs[pid]) for pid in members if pid in seqs])
        msa = run_cmd([tools["mafft"], "--auto", str(fa)], logger)
        a3m.write_text(msa)
        # -name ensures the HMM NAME field matches the subfamily ID so that
        # hhsearch hit output contains correct target IDs for db-search mode.
        run_cmd([tools["hhmake"], "-i", str(a3m), "-o", str(hhm), "-name", subfam], logger)
        return {
            "subfamily_id": subfam,
            "hhm_path": str(hhm),
            "msa_path": str(a3m),
            "n_members_used": len(members),
            "profile_built": True,
            "skipped_reason": "",
            "build_tool": "mafft+hhmake",
            "build_params_json": json.dumps({"mafft": "--auto", "max_members": cap}),
        }

    threads = int(config.get("mmseqs", {}).get("threads", 8))
    # parallel_workers governs concurrent profile build jobs; distinct from
    # per-tool thread counts.  Falls back to profiles.parallel_workers, then
    # to mmseqs.threads for backward compatibility.
    parallel_workers = int(
        config.get("profiles", {}).get("parallel_workers",
        config.get("mmseqs", {}).get("threads", 8))
    )
    rows: list[dict] = []

    agg = (
        smap.group_by("subfamily_id")
        .agg(pl.col("protein_id").sort().head(cap))
        .sort("subfamily_id")
    )
    subfam_members: dict[str, list[str]] = {
        row["subfamily_id"]: row["protein_id"]
        for row in agg.iter_rows(named=True)
    }

    # Determine per-subfamily cluster sizes and identify those to skip
    size_counts: dict[str, int] = {
        row["subfamily_id"]: len(row["protein_id"])
        for row in agg.iter_rows(named=True)
    }

    def _skipped_reason(n: int) -> str:
        if n == 1:
            return "singleton"
        return "below_min_cluster_size"

    # Rows for skipped subfamilies (no profile built)
    skipped_rows: list[dict] = []
    eligible_subfam_members: list[tuple[str, list[str]]] = []
    n_skipped_singletons = 0
    for sfid, members in subfam_members.items():
        n = size_counts.get(sfid, len(members))
        if n < min_cluster_size_for_profile:
            reason = _skipped_reason(n)
            if reason == "singleton":
                n_skipped_singletons += 1
            skipped_rows.append({
                "subfamily_id": sfid,
                "hhm_path": "",
                "msa_path": "",
                "n_members_used": 0,
                "profile_built": False,
                "skipped_reason": reason,
                "build_tool": "",
                "build_params_json": "",
            })
        else:
            eligible_subfam_members.append((sfid, members))

    n_skipped = len(skipped_rows)
    if n_skipped > 0:
        logger.info(
            "Skipping profile build for %d subfamilies (%d singletons) "
            "below min_cluster_size_for_profile=%d",
            n_skipped, n_skipped_singletons, min_cluster_size_for_profile,
        )

    if resume:
        already_built: list[dict] = []
        pending_subfams: list[tuple[str, list[str]]] = []
        for subfam, members in eligible_subfam_members:
            hhm_path = out / f"{subfam}.hhm"
            if hhm_path.exists():
                a3m_path = out / f"{subfam}.a3m"
                already_built.append({
                    "subfamily_id": subfam,
                    "hhm_path": str(hhm_path),
                    "msa_path": str(a3m_path) if a3m_path.exists() else "",
                    "n_members_used": len(members),
                    "profile_built": True,
                    "skipped_reason": "",
                    "build_tool": "mafft+hhmake",
                    "build_params_json": json.dumps({"mafft": "--auto", "max_members": cap}),
                })
            else:
                pending_subfams.append((subfam, members))
        rows.extend(already_built)
        logger.info("Resume: %d profiles already built, building %d missing.",
                    len(already_built), len(pending_subfams))
    else:
        pending_subfams = list(eligible_subfam_members)

    failed: list[str] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
        future_to_subfam: dict[concurrent.futures.Future, str] = {}
        for subfam, members in pending_subfams:
            fut = executor.submit(_process_subfam, subfam, members)
            future_to_subfam[fut] = subfam
        for future in concurrent.futures.as_completed(future_to_subfam):
            subfam = future_to_subfam[future]
            try:
                rows.append(future.result())
            except Exception as exc:
                logger.error("Profile build FAILED for %s: %s", subfam, exc)
                failed.append(subfam)
    if failed:
        logger.warning("%d subfamily profile(s) failed: %s", len(failed), ", ".join(failed[:20]))

    # Combine built + skipped rows; write the full index
    all_rows = rows + skipped_rows
    pl.from_dicts(all_rows).sort("subfamily_id").write_csv(out / "subfamily_profile_index.tsv", separator="\t")
    logger.info(
        "Profile construction complete: %d profiles built/loaded, %d skipped, %d failed",
        len(rows), n_skipped, len(failed),
    )
    return tools


def _run_mmseqs_profile_search(
    outdir: Path,
    idx: pl.DataFrame,
    hhm: dict[str, str],
    lengths: dict[str, int],
    a3m_paths: dict[str, str],
    config: dict,
    logger,
) -> list[dict]:
    """Fast all-vs-all profile-profile search using MMseqs2.

    For large datasets (220K+ families) this is orders of magnitude faster
    than running one ``hhalign`` or ``hhsearch`` per family pair.

    When ``a3m_paths`` contains valid MSA files, builds richer profiles via
    ``mmseqs convertmsa`` + ``mmseqs msa2profile``.  Falls back to an
    iterative sequence search (``--num-iterations 2``) when MSAs are absent.

    Returns a list of raw edge row dicts compatible with ``_HMM_EDGE_SCHEMA``.
    ``prob`` is set to ``0.0`` because MMseqs2 does not produce HH-suite
    probability scores; the existing ``_filter_raw_to_core_relaxed`` function
    handles this correctly via its ``evalue``-based OR condition.
    """
    tools = require_executables(["mmseqs"], config["tools"])
    mmseqs_bin = tools["mmseqs"]
    threads = str(config.get("mmseqs", {}).get("threads", 8))
    topn = int(config["hmm_hmm"]["topN"])
    max_evalue = float(config["hmm_hmm"]["max_evalue_relaxed"])
    tmpdir = _ensure_dir(config.get("mmseqs", {}).get("tmpdir", "tmp/mmseqs"))

    mm_dir = outdir / "mmseqs_profile"
    mm_dir.mkdir(parents=True, exist_ok=True)

    subfam_ids = idx["subfamily_id"].to_list()
    valid_a3m = {s: p for s, p in a3m_paths.items() if p and Path(p).exists()}

    if valid_a3m:
        # Build a combined a3m file where each MSA block starts with the
        # subfamily ID so that MMseqs2 can identify MSAs by subfamily.
        # Blocks are separated by "//" as required by mmseqs convertmsa.
        logger.info(
            "Building combined a3m from %d MSA files for MMseqs2 profile DB",
            len(valid_a3m),
        )
        combined_a3m = mm_dir / "combined.a3m"
        with open(combined_a3m, "w") as fh:
            for subfam in sorted(subfam_ids):
                path = valid_a3m.get(subfam, "")
                if not path:
                    continue
                lines = Path(path).read_text().splitlines()
                first_header_replaced = False
                for line in lines:
                    if not first_header_replaced and line.startswith(">"):
                        fh.write(f">{subfam}\n")
                        first_header_replaced = True
                    elif line:
                        fh.write(line + "\n")
                fh.write("//\n")

        msa_db = mm_dir / "msa_db"
        run_cmd([mmseqs_bin, "convertmsa", str(combined_a3m), str(msa_db)], logger)
        profile_db = mm_dir / "profile_db"
        run_cmd([mmseqs_bin, "msa2profile", str(msa_db), str(profile_db),
                 "--threads", threads], logger)
        query_db = str(profile_db)
        target_db = str(profile_db)
        extra_search_args: list[str] = []
    else:
        # Fallback: extract representative sequences from .hhm files and use
        # iterative sequence search (builds internal profiles on the fly).
        logger.warning(
            "No a3m MSA files found; using representative sequences "
            "for MMseqs2 iterative search (--num-iterations 2)"
        )
        rep_fasta = mm_dir / "reps.faa"
        rep_records: list[FastaRecord] = []
        for subfam in subfam_ids:
            # Extract the clean (ungapped) representative sequence from the
            # first sequence in the .hhm file's associated a3m, or skip.
            path = a3m_paths.get(subfam, "")
            if path and Path(path).exists():
                seq = ""
                past_header = False
                for line in Path(path).read_text().splitlines():
                    if line.startswith(">"):
                        if past_header:
                            break
                        past_header = True
                    elif past_header:
                        # Remove lowercase insertions and gap characters
                        seq += re.sub(r"[a-z.-]", "", line)
                if seq:
                    rep_records.append(FastaRecord(subfam, seq))
        if not rep_records:
            logger.warning("mmseqs-profile: no sequences available, returning empty edges")
            return []
        write_fasta(rep_fasta, rep_records)
        seq_db = mm_dir / "seq_db"
        run_cmd([mmseqs_bin, "createdb", str(rep_fasta), str(seq_db)], logger)
        query_db = str(seq_db)
        target_db = str(seq_db)
        extra_search_args = ["--num-iterations", "2"]

    result_db = mm_dir / "result_db"
    run_cmd(
        [mmseqs_bin, "search",
         query_db, target_db, str(result_db), str(tmpdir),
         "--threads", threads,
         "-e", str(max_evalue),
         "--max-seqs", str(topn + 1),  # +1 because self-hit is always included
         "-s", "7.5",
         *extra_search_args],
        logger,
    )

    result_tsv = mm_dir / "result.tsv"
    run_cmd(
        [mmseqs_bin, "convertalis",
         query_db, target_db, str(result_db), str(result_tsv),
         "--format-output",
         "query,target,pident,alnlen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen"],
        logger,
    )

    if not result_tsv.exists() or result_tsv.stat().st_size == 0:
        logger.warning("MMseqs2 profile search returned no results")
        return []

    cols = ["query", "target", "pident", "alnlen",
            "qstart", "qend", "tstart", "tend",
            "evalue", "bits", "qlen", "tlen"]
    df = pl.read_csv(result_tsv, separator="\t", has_header=False, new_columns=cols)
    df = df.with_columns([
        pl.col("pident").cast(pl.Float64),
        pl.col("alnlen").cast(pl.Int64),
        pl.col("evalue").cast(pl.Float64),
        pl.col("bits").cast(pl.Float64),
        pl.col("qlen").cast(pl.Int64),
        pl.col("tlen").cast(pl.Int64),
    ])

    run_id = "mmseqs_profile_v1"
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    for row in df.iter_rows(named=True):
        q = str(row["query"])
        t = str(row["target"])
        if q == t:
            continue
        if q not in lengths or t not in lengths:
            continue
        canon = tuple(sorted((q, t)))
        if canon in seen:
            # MMseqs2 returns both q→t and t→q; keep the first (highest-scoring)
            continue
        seen.add(canon)
        alnlen = int(row["alnlen"])
        qlen = max(1, int(row["qlen"]))
        tlen = max(1, int(row["tlen"]))
        rows.append({
            "q_subfamily_id": q,
            "t_subfamily_id": t,
            # prob=0.0: MMseqs2 has no HH-suite probability.  The existing
            # _filter_raw_to_core_relaxed function gates on evalue via its OR
            # clause so these rows are correctly filtered.
            "prob": 0.0,
            "evalue": float(row["evalue"]),
            "bits": float(row["bits"]) if row["bits"] is not None else None,
            "qcov": min(1.0, alnlen / qlen),
            "tcov": min(1.0, alnlen / tlen),
            "aln_len": alnlen,
            "pident": float(row["pident"]),
            "tool": "mmseqs_profile",
            "run_id": run_id,
        })

    logger.info(
        "MMseqs2 profile search: %d candidate edges (before HMM-HMM thresholding)",
        len(rows),
    )
    return rows


def hmm_hmm_edges(
    profile_index: str,
    outdir: str,
    config: dict,
    logger,
    candidate_edges_tsv: str | None = None,
    *,
    mode: str | None = None,
    resume: bool = False,
    shard_id: int = 0,
    n_shards: int = 1,
) -> dict[str, str]:
    """Step 3: Compute HMM-HMM profile-profile edges.

    Supports three execution modes selected via *mode* (or ``config["hmm_hmm"]["mode"]``):

    ``pairwise`` (default)
        Runs one ``hhalign`` invocation per candidate pair using a thread pool.

    ``db-search``
        Builds an HH-suite ffindex database from all profiles and runs one
        ``hhsearch`` invocation per unique query subfamily against the whole
        database in parallel.  Results from both the A→B and B→A search
        directions are compared; the best-scoring direction is kept.
        The ffindex files use the ``_hhm`` suffix required by HH-suite 3.x
        (``profiles_db_hhm.ffdata`` / ``profiles_db_hhm.ffindex``).

    ``mmseqs-profile``
        Uses MMseqs2 to run an all-vs-all profile-profile search.  Orders of
        magnitude faster than HH-suite for large datasets (220K+ families).
        Requires only ``mmseqs`` in PATH (already a pipeline dependency).
        When MSA files are available the richer ``convertmsa``/``msa2profile``
        workflow is used; otherwise falls back to iterative sequence search.
    """
    out = _ensure_dir(outdir)
    effective_mode = mode or config.get("hmm_hmm", {}).get("mode", "pairwise")

    if n_shards > 1:
        raw_tsv_path = out / f"hmm_hmm_edges_raw.shard_{shard_id}.tsv"
        progress_path = out / f"hmm_hmm_progress.shard_{shard_id}.ndjson"
    else:
        raw_tsv_path = out / "hmm_hmm_edges_raw.tsv"
        progress_path = out / "hmm_hmm_progress.ndjson"

    idx = pl.read_csv(profile_index, separator="\t").sort("subfamily_id")
    # Filter out subfamilies that were skipped during profile building
    # (hhm_path is empty for singletons / below-threshold clusters).
    if "profile_built" in idx.columns:
        idx = idx.filter(pl.col("profile_built").cast(pl.Boolean))
    else:
        idx = idx.filter(pl.col("hhm_path") != "")
    lengths = dict(zip(idx["subfamily_id"].to_list(), [_parse_hhm_len(p) for p in idx["hhm_path"].to_list()]))
    hhm = dict(zip(idx["subfamily_id"].to_list(), idx["hhm_path"].to_list()))

    cands: list[tuple[str, str]] = []
    topn = int(config["hmm_hmm"]["topN"])
    if candidate_edges_tsv and Path(candidate_edges_tsv).exists():
        cdf = pl.read_csv(candidate_edges_tsv, separator="\t")
        top_cands = (
            cdf.sort("cosine", descending=True)
            .group_by("q_subfamily_id")
            .agg(pl.col("t_subfamily_id").head(topn))
        )
        for row in top_cands.iter_rows(named=True):
            q = row["q_subfamily_id"]
            for t in sorted(row["t_subfamily_id"]):
                cands.append((q, t))
    else:
        subfams = idx["subfamily_id"].to_list()
        n_pairs = len(subfams) * (len(subfams) - 1) // 2
        logger.warning(
            "No candidate edges provided -- using ALL-vs-ALL (%d pairs for %d subfamilies). "
            "This is slow for large datasets; consider running embed+knn first.",
            n_pairs, len(subfams),
        )
        for i, q in enumerate(subfams):
            for t in subfams[i + 1:]:
                cands.append((q, t))

    unique_cands = sorted(set(tuple(sorted(x)) for x in cands))

    if n_shards > 1:
        unique_cands = [p for i, p in enumerate(unique_cands) if i % n_shards == shard_id]
        logger.info("Shard %d/%d: %d candidate pairs", shard_id, n_shards, len(unique_cands))

    processed: set[tuple[str, str]] = set()
    if resume:
        processed = _load_hmm_progress(progress_path)
        logger.info("Resume: %d already-processed pairs loaded from %s", len(processed), str(progress_path))

    pending_cands = [p for p in unique_cands if p not in processed]
    logger.info(
        "Running HMM-HMM (%s mode) on %d candidate pairs (%d skipped by resume)",
        effective_mode, len(pending_cands), len(unique_cands) - len(pending_cands),
    )

    run_id = "hmm_hmm_v2"
    prog_lock = threading.Lock()

    def _make_row(q: str, t: str, m: dict, tool_name: str) -> dict:
        qlen = max(1, lengths.get(q, 0))
        tlen = max(1, lengths.get(t, 0))
        aln_len = int(m["aln_len"])
        qcov = min(1.0, aln_len / qlen)
        tcov = min(1.0, aln_len / tlen)
        return {
            "q_subfamily_id": q,
            "t_subfamily_id": t,
            "prob": float(m["prob"]),
            "evalue": float(m["evalue"]),
            "bits": None,
            "qcov": qcov,
            "tcov": tcov,
            "aln_len": aln_len,
            "pident": float(m["pident"]) if not np.isnan(m["pident"]) else None,
            "tool": tool_name,
            "run_id": run_id,
        }

    threads = int(config.get("mmseqs", {}).get("threads", 8))
    # parallel_workers: concurrent hhalign/hhsearch jobs.  Distinct from
    # per-tool thread counts.  Falls back to hmm_hmm.parallel_workers, then
    # mmseqs.threads for backward compatibility.
    parallel_workers = int(
        config.get("hmm_hmm", {}).get("parallel_workers",
        config.get("mmseqs", {}).get("threads", 8))
    )

    # ------------------------------------------------------------------
    # Pairwise mode
    # ------------------------------------------------------------------
    if effective_mode == "pairwise":
        tools = require_executables(["hhalign"], config["tools"])
        raw_rows: list[dict] = []
        failed_edges: list[str] = []

        def _process_hh_edge(q: str, t: str) -> dict:
            out_hhr = out / f"{q}__{t}.hhr"
            run_cmd([tools["hhalign"], "-i", hhm[q], "-t", hhm[t], "-o", str(out_hhr)], logger)
            return _make_row(q, t, _parse_hhr_metrics(out_hhr), "hhalign")

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_pair: dict[concurrent.futures.Future, tuple[str, str]] = {}
            for q, t in pending_cands:
                fut = executor.submit(_process_hh_edge, q, t)
                future_to_pair[fut] = (q, t)
            for future in concurrent.futures.as_completed(future_to_pair):
                pair = future_to_pair[future]
                try:
                    row = future.result()
                    raw_rows.append(row)
                    _append_hmm_progress(
                        progress_path,
                        {"ts": time.time(), "q": pair[0], "t": pair[1], "status": "ok",
                         "prob": row["prob"], "evalue": row["evalue"], "aln_len": row["aln_len"]},
                        prog_lock,
                    )
                except Exception as exc:
                    logger.error("hhalign FAILED for %s vs %s: %s", pair[0], pair[1], exc)
                    failed_edges.append(f"{pair[0]}__{pair[1]}")
                    _append_hmm_progress(
                        progress_path,
                        {"ts": time.time(), "q": pair[0], "t": pair[1], "status": "failed", "error": str(exc)},
                        prog_lock,
                    )

        if failed_edges:
            logger.warning("%d hhalign edge(s) failed: %s", len(failed_edges), ", ".join(failed_edges[:20]))

        if resume and raw_tsv_path.exists() and processed:
            existing = pl.read_csv(raw_tsv_path, separator="\t", null_values=["NA"], schema_overrides=_HMM_EDGE_SCHEMA)
            new_df = pl.from_dicts(raw_rows, schema=_HMM_EDGE_SCHEMA) if raw_rows else pl.DataFrame(schema=_HMM_EDGE_SCHEMA)
            pl.concat([existing, new_df]).sort(["q_subfamily_id", "t_subfamily_id"]).write_csv(
                raw_tsv_path, separator="\t", null_value="NA"
            )
        else:
            raw_df = (
                pl.from_dicts(raw_rows, schema=_HMM_EDGE_SCHEMA).sort(["q_subfamily_id", "t_subfamily_id"])
                if raw_rows
                else pl.DataFrame(schema=_HMM_EDGE_SCHEMA)
            )
            raw_df.write_csv(raw_tsv_path, separator="\t", null_value="NA")

    # ------------------------------------------------------------------
    # DB-search mode (hhsearch against ffindex database, parallelised)
    # ------------------------------------------------------------------
    elif effective_mode == "db-search":
        try:
            db_tools = require_executables(["hhsearch", "ffindex_build", "cstranslate"], config["tools"])
        except RuntimeError as exc:
            logger.warning(
                "db-search mode requested but tools unavailable (%s); falling back to pairwise hhalign.", exc
            )
            return hmm_hmm_edges(
                profile_index, outdir, config, logger, candidate_edges_tsv,
                mode="pairwise", resume=resume, shard_id=shard_id, n_shards=n_shards,
            )

        tools = {**db_tools}

        # HH-suite 3.x hhsearch expects the profile database files to carry the
        # "_hhm" and "_a3m" suffixes: profiles_db_hhm.ffdata / profiles_db_hhm.ffindex
        # and profiles_db_a3m.ffdata / profiles_db_a3m.ffindex.
        # Passing -d profiles_db to hhsearch resolves these files automatically.
        db_dir = out / "hhsearch_db"
        db_prefix = db_dir / "profiles_db"

        # Read a3m (MSA) paths from the profile index for _a3m database.
        a3m_col = "msa_path" if "msa_path" in idx.columns else None
        a3m_paths_db: dict[str, str] = {}
        if a3m_col:
            a3m_paths_db = dict(zip(
                idx["subfamily_id"].to_list(),
                idx[a3m_col].to_list(),
            ))

        _build_hhsuite_db(
            db_dir, db_prefix, hhm, a3m_paths_db,
            db_tools["ffindex_build"], db_tools["cstranslate"], logger,
        )

        query_to_targets: dict[str, set[str]] = defaultdict(set)
        for q, t in pending_cands:
            query_to_targets[q].add(t)
            query_to_targets[t].add(q)

        pending_queries = sorted(
            q for q in query_to_targets
            if any((q, t) not in processed and (t, q) not in processed for t in query_to_targets[q])
        )

        logger.info("Running hhsearch (parallel, %d workers, %d threads/query) for %d unique query profiles",
                    parallel_workers, threads, len(pending_queries))

        raw_rows_db: list[dict] = []
        failed_queries: list[str] = []
        # Track the best result per canonical pair across both search directions.
        # HH-suite alignments are asymmetric: searching A→B can give different
        # scores than B→A.  We run both and keep whichever has the highest prob.
        best_db_results: dict[tuple[str, str], dict] = {}
        best_db_lock = threading.Lock()

        def _do_hhsearch(q: str) -> tuple[str, list[dict], Exception | None]:
            q_hhr = out / f"{q}__hhsearch.hhr"
            try:
                run_cmd(
                    [db_tools["hhsearch"], "-i", hhm[q], "-d", str(db_prefix),
                     "-o", str(q_hhr), "-cpu", str(threads)],
                    logger,
                )
                return q, _parse_hhr_all_hits(q_hhr), None
            except Exception as exc:
                return q, [], exc

        with concurrent.futures.ThreadPoolExecutor(max_workers=parallel_workers) as executor:
            future_to_q: dict[concurrent.futures.Future, str] = {
                executor.submit(_do_hhsearch, q): q for q in pending_queries
            }
            for future in concurrent.futures.as_completed(future_to_q):
                q, hits, exc = future.result()
                if exc is not None:
                    logger.error("hhsearch FAILED for query %s: %s", q, exc)
                    failed_queries.append(q)
                    _append_hmm_progress(
                        progress_path,
                        {"ts": time.time(), "q": q, "t": "", "status": "failed", "error": str(exc)},
                        prog_lock,
                    )
                    continue
                for hit in hits:
                    t = hit["target_id"]
                    if t == q or t not in hhm:
                        continue
                    if (t not in query_to_targets.get(q, set())
                            and q not in query_to_targets.get(t, set())):
                        continue
                    canon = tuple(sorted((q, t)))
                    if canon in processed:
                        continue
                    row = _make_row(q, t, hit, "hhsearch")
                    with best_db_lock:
                        existing = best_db_results.get(canon)
                        if existing is None:
                            # First time we see this pair — record it immediately so
                            # crash recovery can resume from this point.
                            best_db_results[canon] = row
                            _append_hmm_progress(
                                progress_path,
                                {"ts": time.time(), "q": canon[0], "t": canon[1],
                                 "status": "ok", "prob": row["prob"],
                                 "evalue": row["evalue"], "aln_len": row["aln_len"]},
                                prog_lock,
                            )
                        elif (row["prob"] > existing["prob"]
                              or (row["prob"] == existing["prob"]
                                  and row["evalue"] < existing["evalue"])):
                            # Better result found from the reverse direction — update
                            # the in-memory dict (progress log keeps the first entry,
                            # but the final TSV will have the best score).
                            best_db_results[canon] = row

        if failed_queries:
            logger.warning("%d hhsearch query/queries failed: %s",
                           len(failed_queries), ", ".join(failed_queries[:20]))

        raw_rows_db = list(best_db_results.values())

        if resume and raw_tsv_path.exists() and processed:
            existing_df = pl.read_csv(raw_tsv_path, separator="\t", null_values=["NA"],
                                      schema_overrides=_HMM_EDGE_SCHEMA)
            new_df = (pl.from_dicts(raw_rows_db, schema=_HMM_EDGE_SCHEMA)
                      if raw_rows_db else pl.DataFrame(schema=_HMM_EDGE_SCHEMA))
            pl.concat([existing_df, new_df]).sort(["q_subfamily_id", "t_subfamily_id"]).write_csv(
                raw_tsv_path, separator="\t", null_value="NA"
            )
        else:
            raw_df = (
                pl.from_dicts(raw_rows_db, schema=_HMM_EDGE_SCHEMA).sort(["q_subfamily_id", "t_subfamily_id"])
                if raw_rows_db
                else pl.DataFrame(schema=_HMM_EDGE_SCHEMA)
            )
            raw_df.write_csv(raw_tsv_path, separator="\t", null_value="NA")

    # ------------------------------------------------------------------
    # MMseqs2 profile-profile search mode (fast, scales to 220K+ families)
    # ------------------------------------------------------------------
    else:
        # effective_mode == "mmseqs-profile"
        tools = require_executables(["mmseqs"], config["tools"])
        a3m_col_mp = "msa_path" if "msa_path" in idx.columns else None
        a3m_paths_mp: dict[str, str] = {}
        if a3m_col_mp:
            a3m_paths_mp = dict(zip(
                idx["subfamily_id"].to_list(),
                idx[a3m_col_mp].to_list(),
            ))

        raw_rows_mm = _run_mmseqs_profile_search(
            out, idx, hhm, lengths, a3m_paths_mp, config, logger,
        )

        if resume and raw_tsv_path.exists():
            existing_df = pl.read_csv(raw_tsv_path, separator="\t", null_values=["NA"],
                                      schema_overrides=_HMM_EDGE_SCHEMA)
            new_df = (pl.from_dicts(raw_rows_mm, schema=_HMM_EDGE_SCHEMA)
                      if raw_rows_mm else pl.DataFrame(schema=_HMM_EDGE_SCHEMA))
            pl.concat([existing_df, new_df]).sort(["q_subfamily_id", "t_subfamily_id"]).write_csv(
                raw_tsv_path, separator="\t", null_value="NA"
            )
        else:
            raw_df = (
                pl.from_dicts(raw_rows_mm, schema=_HMM_EDGE_SCHEMA).sort(["q_subfamily_id", "t_subfamily_id"])
                if raw_rows_mm
                else pl.DataFrame(schema=_HMM_EDGE_SCHEMA)
            )
            raw_df.write_csv(raw_tsv_path, separator="\t", null_value="NA")

        # Write individual pair progress entries for resume support
        for row in raw_rows_mm:
            _append_hmm_progress(
                progress_path,
                {"ts": time.time(), "q": row["q_subfamily_id"], "t": row["t_subfamily_id"],
                 "status": "ok", "prob": row["prob"],
                 "evalue": row["evalue"], "aln_len": row["aln_len"]},
                prog_lock,
            )

    if n_shards == 1:
        raw = pl.read_csv(raw_tsv_path, separator="\t", null_values=["NA"], schema_overrides=_HMM_EDGE_SCHEMA)
        _filter_raw_to_core_relaxed(raw, config["hmm_hmm"], out)

    return tools


def embed(reps_fasta: str, outdir: str, config: dict, weights_path: str | None, logger, resume: bool = False) -> None:
    """Step 4: Generate ESM-2 mean-pooled embeddings for subfamily representatives.

    Supports mid-run resumability via per-batch checkpoints.  If
    ``embed.checkpoint_dir`` is set (non-empty), each completed batch is
    persisted as a ``.npy`` file under that directory.  On a subsequent call
    with ``resume=True`` the function reloads any already-computed batches and
    skips them, so an OOM or timeout mid-run does not require restarting from
    scratch.
    """
    out_path = Path(outdir) / "embeddings.npy"
    if resume and out_path.exists():
        logger.info("Resume: embeddings already exist at %s, skipping.", str(out_path))
        return
    try:
        import esm
        import torch
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("Embedding requires PyTorch + fair-esm") from exc

    if not weights_path:
        raise RuntimeError("Offline mode: --weights_path is required (no silent downloads).")
    if not Path(weights_path).exists():
        raise RuntimeError(f"weights_path does not exist: {weights_path}")

    out = _ensure_dir(outdir)

    # ------------------------------------------------------------------ #
    # Checkpoint directory for mid-run resumability                       #
    # ------------------------------------------------------------------ #
    ckpt_dir_str = str(config["embed"].get("checkpoint_dir", ""))
    ckpt_dir: Path | None = _ensure_dir(ckpt_dir_str) if ckpt_dir_str else None

    recs = sorted(read_fasta(reps_fasta), key=lambda x: x.id)

    import functools
    _orig_torch_load = torch.load
    torch.load = functools.partial(_orig_torch_load, weights_only=False)
    try:
        logger.info("Loading ESM model from %s", weights_path)
        model, alphabet = esm.pretrained.load_model_and_alphabet_local(weights_path)
    finally:
        torch.load = _orig_torch_load
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    device_str = str(config["embed"].get("device", "cpu"))
    device = torch.device(device_str)
    model = model.to(device)
    logger.info("Embedding device: %s", device)

    max_len = int(config["embed"]["max_len"])
    policy = config["embed"]["long_seq_policy"]
    batch_size = int(config["embed"]["batch_size"])
    seqs = []
    embedded_recs = []
    n_skipped_long = 0
    n_skipped_empty = 0
    for r in recs:
        if len(r.seq) == 0:
            logger.warning("Skipping %s: sequence is empty after stripping stop codons.", r.id)
            n_skipped_empty += 1
            continue
        if len(r.seq) > max_len and policy == "skip":
            n_skipped_long += 1
            continue
        seq = r.seq[:max_len] if len(r.seq) > max_len and policy == "truncate" else r.seq
        seqs.append((r.id, seq))
        embedded_recs.append(r)

    if n_skipped_long:
        logger.warning("Skipped %d sequences longer than max_len=%d (long_seq_policy='skip').",
                       n_skipped_long, max_len)
    if not seqs:
        raise RuntimeError(
            f"No sequences remain to embed after filtering "
            f"(total={len(recs)}, skipped_empty={n_skipped_empty}, skipped_long={n_skipped_long}). "
            "Check your FASTA file and long_seq_policy / max_len settings."
        )

    total_batches = (len(seqs) + batch_size - 1) // batch_size

    # Check for already-completed checkpoints when resuming
    embs: list[np.ndarray] = []
    start_batch = 0
    if resume and ckpt_dir is not None:
        for batch_num in range(total_batches):
            ckpt_file = ckpt_dir / f"embed_batch_{batch_num:06d}.npy"
            if ckpt_file.exists():
                try:
                    loaded = np.load(str(ckpt_file))
                    embs.append(loaded)
                    start_batch = batch_num + 1
                except Exception as exc:
                    logger.warning("Could not load checkpoint %s: %s", str(ckpt_file), exc)
                    break
            else:
                break
        if start_batch > 0:
            logger.info(
                "Resume: loaded %d completed embedding batches from %s, continuing from batch %d/%d",
                start_batch, str(ckpt_dir), start_batch + 1, total_batches,
            )

    logger.info(
        "Processing %d sequences in %d batches of size %d (starting from batch %d)",
        len(seqs), total_batches, batch_size, start_batch + 1,
    )

    for i in range(start_batch * batch_size, len(seqs), batch_size):
        batch = seqs[i:i + batch_size]
        batch_num = i // batch_size
        logger.info("Processing batch %d/%d (%d sequences)", batch_num + 1, total_batches, len(batch))
        labels, strs, toks = batch_converter(batch)
        toks = toks.to(device)
        with torch.no_grad():
            outp = model(toks, repr_layers=[model.num_layers], return_contacts=False)
        reps = outp["representations"][model.num_layers]
        batch_embs = np.vstack([
            reps[j, 1: len(seq) + 1].mean(0).cpu().numpy()
            for j, seq in enumerate(strs)
        ]).astype(np.float32)
        embs.append(batch_embs)

        if ckpt_dir is not None:
            ckpt_file = ckpt_dir / f"embed_batch_{batch_num:06d}.npy"
            np.save(str(ckpt_file), batch_embs)

        del toks, outp, reps
        if device.type == "cuda":
            torch.cuda.empty_cache()

    mat = np.vstack(embs).astype(np.float32)
    np.save(out / "embeddings.npy", mat)
    (out / "ids.txt").write_text("\n".join([r.id for r in embedded_recs]) + "\n")
    pl.DataFrame({
        "subfamily_id": [r.id for r in embedded_recs],
        "rep_length_aa": [len(r.seq) for r in embedded_recs],
    }).write_csv(out / "lengths.tsv", separator="\t")
    (out / "metadata.json").write_text(
        json.dumps({
            "model_name": config["embed"]["esm_model_name"],
            "weights_path": str(weights_path),
            "pooling": config["embed"]["pooling"],
            "long_seq_policy": policy,
            "max_len": max_len,
            "n_subfamilies": len(embedded_recs),
            "dim": int(mat.shape[1]),
        }, indent=2)
    )

    # Clean up checkpoints now that the final file is written
    if ckpt_dir is not None:
        for batch_num in range(total_batches):
            ckpt_file = ckpt_dir / f"embed_batch_{batch_num:06d}.npy"
            ckpt_file.unlink(missing_ok=True)


def knn(
    embeddings_npy: str, ids_txt: str, lengths_tsv: str, out_tsv: str,
    config: dict, logger=None, resume: bool = False,
    subfamily_map: str | None = None,
) -> None:
    """Step 5: Find K-nearest neighbors in embedding space."""
    if resume and Path(out_tsv).exists():
        if logger:
            logger.info("Resume: KNN edges already exist at %s, skipping.", out_tsv)
        return
    X = np.load(embeddings_npy)
    ids = [x.strip() for x in Path(ids_txt).read_text().splitlines() if x.strip()]
    _lens_df = pl.read_csv(lengths_tsv, separator="\t")
    lens = dict(zip(_lens_df["subfamily_id"].to_list(), _lens_df["rep_length_aa"].to_list()))
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)

    mode = str(config["knn"].get("mode", "knn"))
    device = str(config["knn"].get("device", "cpu"))

    if logger:
        _has_faiss_gpu = False
        _has_torch_cuda = False
        try:
            import faiss  # type: ignore
            _has_faiss_gpu = hasattr(faiss, "StandardGpuResources")
        except Exception:
            pass
        try:
            import torch
            _has_torch_cuda = torch.cuda.is_available()
        except Exception:
            pass
        logger.info("KNN mode: %s, device: %s (FAISS-GPU: %s, PyTorch CUDA: %s)",
                     mode, device,
                     "available" if _has_faiss_gpu else "not available",
                     "available" if _has_torch_cuda else "not available")

    if mode == "rkcnn":
        from .rkcnn import rkcnn_candidate_edges

        # Build integer labels from subfamily IDs.  Each embedding row is a
        # subfamily representative — its own ID becomes its class label.
        # In this "one-sample-per-class" regime rKCNN still works: it finds
        # which *other* subfamilies are most similar by evaluating conditional
        # proximity across random subspaces.  The separation score measures
        # how well each subspace distinguishes the overall class structure,
        # not individual classes.
        id_to_int = {sid: i for i, sid in enumerate(sorted(set(ids)))}
        labels = np.array([id_to_int[sid] for sid in ids])

        rows = rkcnn_candidate_edges(X, ids, lens, labels, config, logger_=logger)
    else:
        # Standard KNN mode
        k = max(1, min(int(config["knn"]["k"]), len(ids) - 1))
        try:
            from .rkcnn import _build_faiss_index
            index = _build_faiss_index(X.astype(np.float32), device)
            sims, nbrs = index.search(X.astype(np.float32), k + 1)
        except Exception:
            if device.startswith("cuda") and logger:
                logger.warning("FAISS-GPU not available; falling back to CPU.")
            from sklearn.neighbors import NearestNeighbors
            nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine")
            nn.fit(X)
            d, nbrs = nn.kneighbors(X)
            sims = 1.0 - d

        rows = []
        for i, q in enumerate(ids):
            for j, nidx in enumerate(nbrs[i][1:]):
                t = ids[nidx]
                cosine = float(sims[i][j + 1])
                qlen = int(lens[q])
                tlen = int(lens[t])
                ratio = qlen / max(1, tlen)
                pass_lr = int(float(config["knn"]["min_len_ratio"]) <= ratio <= float(config["knn"]["max_len_ratio"]))
                if cosine >= float(config["knn"]["min_cosine"]) and pass_lr:
                    rows.append({
                        "q_subfamily_id": q, "t_subfamily_id": t, "cosine": cosine,
                        "q_len": qlen, "t_len": tlen, "len_ratio": ratio, "pass_len_ratio": pass_lr,
                    })

    Path(out_tsv).parent.mkdir(parents=True, exist_ok=True)
    if rows:
        pl.from_dicts(rows).sort(["q_subfamily_id", "t_subfamily_id"]).write_csv(out_tsv, separator="\t")
    else:
        pl.DataFrame(schema={
            "q_subfamily_id": pl.String, "t_subfamily_id": pl.String, "cosine": pl.Float64,
            "q_len": pl.Int64, "t_len": pl.Int64, "len_ratio": pl.Float64, "pass_len_ratio": pl.Int32,
        }).write_csv(out_tsv, separator="\t")


def merge_graph(
    hmm_core_tsv: str,
    emb_tsv: str,
    out_strict_tsv: str,
    out_functional_tsv: str,
    config: dict,
    hmm_relaxed_tsv: str | None = None,
    logger=None,
    resume: bool = False,
) -> None:
    """Step 6: Merge HMM and embedding edges into strict and functional graphs."""
    if resume and Path(out_strict_tsv).exists() and Path(out_functional_tsv).exists():
        if logger:
            logger.info("Resume: merged graph files already exist, skipping merge-graph.")
        return
    Path(out_strict_tsv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_functional_tsv).parent.mkdir(parents=True, exist_ok=True)

    _hmm_schema = {"q_subfamily_id": pl.String, "t_subfamily_id": pl.String, "edge_weight": pl.Float64}
    _emb_schema = {"q_subfamily_id": pl.String, "t_subfamily_id": pl.String, "cosine": pl.Float64}

    core = pl.read_csv(hmm_core_tsv, separator="\t") if Path(hmm_core_tsv).exists() else pl.DataFrame(schema=_hmm_schema)
    emb = pl.read_csv(emb_tsv, separator="\t") if Path(emb_tsv).exists() else pl.DataFrame(schema=_emb_schema)
    relaxed = (
        pl.read_csv(hmm_relaxed_tsv, separator="\t")
        if hmm_relaxed_tsv and Path(hmm_relaxed_tsv).exists()
        else pl.DataFrame(schema=_hmm_schema)
    )

    if len(core) > 0:
        strict = (
            core.rename({"q_subfamily_id": "qid", "t_subfamily_id": "tid"})
            .with_columns(pl.col("edge_weight").alias("weight"), pl.lit("hmm_core").alias("source"))
            .select(["qid", "tid", "weight", "source"])
            .sort(["qid", "tid"])
        )
    else:
        strict = pl.DataFrame(schema={"qid": pl.String, "tid": pl.String, "weight": pl.Float64, "source": pl.String})
    strict.write_csv(out_strict_tsv, separator="\t")

    policy = config["graph"]["edge_weight_policy"]
    edges: dict[tuple[str, str], dict[str, float | str]] = {}

    for df, src in [(core, "hmm_core"), (relaxed, "hmm_relaxed")]:
        for row in df.iter_rows(named=True):
            k = tuple(sorted((row["q_subfamily_id"], row["t_subfamily_id"])))
            edges[k] = {"hmm": float(row["edge_weight"]), "emb": 0.0, "source": src}

    for row in emb.iter_rows(named=True):
        k = tuple(sorted((row["q_subfamily_id"], row["t_subfamily_id"])))
        if policy == "strict":
            continue
        if policy == "gated":
            weak = edges.get(k, {"hmm": 0.0})["hmm"]
            if float(weak) < float(config["graph"]["weak_hmm_threshold"]):
                continue
        if k not in edges:
            edges[k] = {"hmm": 0.0, "emb": float(row["cosine"]), "source": "emb"}
        else:
            edges[k]["emb"] = max(float(edges[k]["emb"]), float(row["cosine"]))
            edges[k]["source"] = "both"

    rows = []
    for (q, t), w in edges.items():
        hmm_w = float(w["hmm"])
        emb_w = float(w["emb"])
        if policy == "union":
            weight = max(hmm_w, emb_w)
        elif policy == "downweight_embeddings":
            weight = float(config["graph"]["w_hmm"]) * hmm_w + float(config["graph"]["w_emb"]) * emb_w
        else:
            weight = hmm_w
        rows.append({"qid": q, "tid": t, "weight": weight, "source": str(w["source"]),
                     "hmm_weight": hmm_w, "emb_weight": emb_w})

    if rows:
        pl.from_dicts(rows).sort(["qid", "tid"]).write_csv(out_functional_tsv, separator="\t")
    else:
        pl.DataFrame(schema={
            "qid": pl.String, "tid": pl.String, "weight": pl.Float64,
            "source": pl.String, "hmm_weight": pl.Float64, "emb_weight": pl.Float64,
        }).write_csv(out_functional_tsv, separator="\t")


def _cluster_leiden(edges: pl.DataFrame, resolution: float, seed: int) -> pl.DataFrame:
    import igraph as ig
    import leidenalg

    nodes = sorted(set(edges["qid"].to_list()).union(set(edges["tid"].to_list())))
    n2i = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(
        n=len(nodes),
        edges=[(n2i[a], n2i[b]) for a, b in edges.select(["qid", "tid"]).iter_rows()],
        directed=False,
    )
    g.es["weight"] = edges["weight"].to_list()
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=resolution,
        seed=seed,
    )
    return pl.DataFrame({
        "subfamily_id": pl.Series(name="subfamily_id", values=nodes, dtype=pl.String),
        "community": pl.Series(name="community", values=part.membership, dtype=pl.Int64),
    })


def cluster_families(
    merged_edges_strict: str,
    merged_edges_functional: str,
    subfamily_map: str,
    outdir: str,
    config: dict,
    method: str = "leiden",
    logger=None,
    resume: bool = False,
) -> None:
    out = _ensure_dir(outdir)
    if resume and (out / "subfamily_to_family_strict.tsv").exists() and (out / "subfamily_to_family_functional.tsv").exists():
        if logger:
            logger.info("Resume: cluster-families outputs already exist in %s, skipping.", str(out))
        return
    if method == "mcl":
        require_executables(["mcl"], config["tools"])
        raise RuntimeError("MCL comparison requested but not implemented in this minimal release.")

    strict_edges = pl.read_csv(merged_edges_strict, separator="\t")
    func_edges = pl.read_csv(merged_edges_functional, separator="\t")

    strict = _cluster_leiden(strict_edges, float(config["graph"]["leiden_resolution_strict"]), int(config["graph"]["seed"]))
    func = _cluster_leiden(func_edges, float(config["graph"]["leiden_resolution_functional"]), int(config["graph"]["seed"]))

    strict = strict.with_columns(
        pl.col("community").map_elements(lambda x: f"famS_{int(x):06d}", return_dtype=pl.String).alias("family_id")
    )
    func = func.with_columns(
        pl.col("community").map_elements(lambda x: f"famF_{int(x):06d}", return_dtype=pl.String).alias("family_id")
    )

    # ------------------------------------------------------------------ #
    # Preserve edge-less subfamilies as singleton families                #
    # Every subfamily in subfamily_map.tsv must appear in the output.    #
    # Subfamilies that have no retained graph edges are not assigned a   #
    # community by Leiden, so we give each one a unique singleton family. #
    # ------------------------------------------------------------------ #
    smap = pl.read_csv(subfamily_map, separator="\t")
    all_subfamilies: list[str] = sorted(smap["subfamily_id"].unique().to_list())

    assigned_strict: set[str] = set(strict["subfamily_id"].to_list())
    assigned_func: set[str] = set(func["subfamily_id"].to_list())

    # Determine the next available community index for each tier so that
    # singleton IDs never collide with graph-derived IDs.
    next_strict_idx = (strict["community"].max() + 1) if len(strict) > 0 else 0
    next_func_idx = (func["community"].max() + 1) if len(func) > 0 else 0

    singleton_strict_rows: list[dict] = []
    singleton_func_rows: list[dict] = []
    for subfam in all_subfamilies:
        if subfam not in assigned_strict:
            singleton_strict_rows.append({
                "subfamily_id": subfam,
                "community": next_strict_idx,
                "family_id": f"famS_{next_strict_idx:06d}",
            })
            next_strict_idx += 1
        if subfam not in assigned_func:
            singleton_func_rows.append({
                "subfamily_id": subfam,
                "community": next_func_idx,
                "family_id": f"famF_{next_func_idx:06d}",
            })
            next_func_idx += 1

    if singleton_strict_rows:
        strict = pl.concat([
            strict,
            pl.DataFrame(singleton_strict_rows),
        ]).sort("subfamily_id")
        if logger:
            logger.info(
                "cluster-families (strict): %d edge-less subfamilies assigned singleton family IDs",
                len(singleton_strict_rows),
            )
    if singleton_func_rows:
        func = pl.concat([
            func,
            pl.DataFrame(singleton_func_rows),
        ]).sort("subfamily_id")
        if logger:
            logger.info(
                "cluster-families (functional): %d edge-less subfamilies assigned singleton family IDs",
                len(singleton_func_rows),
            )

    strict_out = strict.select(["subfamily_id", "family_id"]).with_columns(
        pl.lit(method).alias("method"),
        pl.lit(json.dumps({"resolution": config["graph"]["leiden_resolution_strict"], "seed": config["graph"]["seed"]})).alias("method_params_json"),
        pl.lit("Mode A strict core-domain families").alias("notes"),
    )
    strict_out.write_csv(out / "subfamily_to_family_strict.tsv", separator="\t")
    strict_out.write_csv(out / "subfamily_to_family.tsv", separator="\t")

    func_out = func.select(["subfamily_id", "family_id"]).with_columns(
        pl.lit(method).alias("method"),
        pl.lit(json.dumps({"resolution": config["graph"]["leiden_resolution_functional"], "seed": config["graph"]["seed"]})).alias("method_params_json"),
        pl.lit("Mode B functional neighborhoods").alias("notes"),
    )
    func_out.write_csv(out / "subfamily_to_family_functional.tsv", separator="\t")

    for tag, sdf in [("strict", strict_out), ("functional", func_out)]:
        st = sdf.group_by("family_id").agg(pl.col("subfamily_id").len().alias("n_subfamilies"))
        prot = (
            smap.join(sdf.select(["subfamily_id", "family_id"]), on="subfamily_id", how="left")
            .group_by("family_id")
            .agg(pl.col("protein_id").n_unique().alias("n_proteins"))
        )
        stats = (
            st.join(prot, on="family_id", how="left")
            .with_columns(
                pl.lit(None).cast(pl.Float64).alias("edge_density_core"),
                pl.lit(None).cast(pl.Float64).alias("mean_rep_len"),
                pl.lit("{}").alias("edge_source_counts_json"),
            )
            .sort("family_id")
        )
        stats.write_csv(out / f"family_stats_{tag}.tsv", separator="\t")


def map_proteins_to_families(
    proteins_fasta: str,
    subfamily_to_family_strict: str,
    subfamily_to_family_functional: str,
    subfamily_map: str,
    outdir: str,
    config: dict,
    logger=None,
    resume: bool = False,
) -> None:
    if logger is None:
        import logging
        logger = logging.getLogger(__name__)
    out = _ensure_dir(outdir)
    if resume and (out / "protein_vs_profile_hits.tsv").exists():
        logger.info("Resume: map-proteins-to-families outputs already exist in %s, skipping.", str(out))
        return

    tools = require_executables(["mmseqs"], config["tools"])
    sub_reps = Path(outdir).parent / "01_mmseqs" / "subfamily_reps.faa"
    if not sub_reps.exists():
        raise RuntimeError(f"subfamily_reps.faa not found at {sub_reps}")

    db_out = out / "mmseqs_search"
    db_out.mkdir(parents=True, exist_ok=True)
    res_tsv = db_out / "search_results.tsv"
    tmpdir = _ensure_dir(config.get("mmseqs", {}).get("tmpdir", "tmp/mmseqs"))
    threads = str(config.get("mmseqs", {}).get("threads", 8))

    mmseqs_cmd = [
        tools["mmseqs"], "easy-search",
        proteins_fasta, str(sub_reps), str(res_tsv), str(tmpdir),
        "--threads", threads,
        "-e", str(config["mapping"]["max_evalue"]),
        "--format-output", "query,target,pident,alnlen,mismatch,gapopen,qstart,qend,tstart,tend,evalue,bits,qlen,tlen",
    ]
    run_cmd(mmseqs_cmd, logger)

    _s2f_df = pl.read_csv(subfamily_to_family_strict, separator="\t")
    s2f = dict(zip(_s2f_df["subfamily_id"].to_list(), _s2f_df["family_id"].to_list()))
    _f2f_df = pl.read_csv(subfamily_to_family_functional, separator="\t")
    f2f = dict(zip(_f2f_df["subfamily_id"].to_list(), _f2f_df["family_id"].to_list()))

    _cols = ["query", "target", "pident", "alnlen", "mismatch", "gapopen",
             "qstart", "qend", "tstart", "tend", "evalue", "bits", "qlen", "tlen"]
    if res_tsv.stat().st_size > 0:
        df = pl.read_csv(res_tsv, separator="\t", has_header=False, new_columns=_cols)
    else:
        df = pl.DataFrame(schema={c: pl.String for c in _cols})

    import warnings
    min_cov = float(config["mapping"]["profile_cov_min"])
    # Support both min_pident (canonical) and legacy min_prob
    mapping_cfg = config["mapping"]
    if "min_prob" in mapping_cfg and "min_pident" not in mapping_cfg:
        warnings.warn(
            "Config key 'mapping.min_prob' is deprecated. Use 'mapping.min_pident' instead.",
            DeprecationWarning,
            stacklevel=2,
        )
        min_pident = float(mapping_cfg["min_prob"])
    else:
        min_pident = float(mapping_cfg.get("min_pident", 0.0))
    min_seg_len = int(config["mapping"].get("min_segment_len", 0))
    if len(df) > 0:
        df = df.with_columns([
            pl.col("pident").cast(pl.Float64),
            pl.col("alnlen").cast(pl.Int64),
            pl.col("qstart").cast(pl.Int64),
            pl.col("qend").cast(pl.Int64),
            pl.col("tstart").cast(pl.Int64),
            pl.col("tend").cast(pl.Int64),
            pl.col("evalue").cast(pl.Float64),
            pl.col("bits").cast(pl.Float64),
            pl.col("qlen").cast(pl.Int64),
            pl.col("tlen").cast(pl.Int64),
        ]).with_columns([
            (pl.col("alnlen") / pl.col("tlen")).alias("profile_cov"),
            (pl.col("alnlen") / pl.col("qlen")).alias("protein_cov"),
        ])
        pre_count = len(df)
        df = df.filter(pl.col("profile_cov") >= min_cov)
        if min_pident > 0:
            df = df.filter(pl.col("pident") >= min_pident)
        if min_seg_len > 0:
            df = df.filter(pl.col("alnlen") >= min_seg_len)
        if logger:
            logger.info("Mapping filter: %d -> %d hits (min_cov=%.2f, min_pident=%.1f, min_seg_len=%d)",
                        pre_count, len(df), min_cov, min_pident, min_seg_len)
        df = df.sort(["query", "evalue", "bits"], descending=[False, False, True])

    max_overlap = int(config["mapping"]["max_overlap_aa"])

    hits = []
    segs = []

    def _overlap(s1, e1, s2, e2):
        return max(0, min(e1, e2) - max(s1, s2))

    if len(df) > 0:
        for grp in df.partition_by("query", maintain_order=True):
            pid = grp["query"][0]
            accepted_intervals = []
            rank = 1
            for row in grp.iter_rows(named=True):
                qs, qe = int(row["qstart"]), int(row["qend"])
                too_much_overlap = any(
                    _overlap(qs, qe, acc_s, acc_e) > max_overlap
                    for (acc_s, acc_e) in accepted_intervals
                )
                if not too_much_overlap:
                    accepted_intervals.append((qs, qe))
                    subfam = row["target"]
                    fam_s = s2f.get(subfam, "")
                    fam_f = f2f.get(subfam, "")
                    hits.append({
                        "protein_id": pid, "profile_id": subfam, "subfamily_id": subfam,
                        "family_id": fam_s if fam_s else "",
                        "start_aa": qs, "end_aa": qe,
                        "prob": row["pident"], "evalue": row["evalue"],
                        "profile_cov": row["profile_cov"], "protein_cov": row["protein_cov"],
                        "aln_len": row["alnlen"], "tool": "mmseqs",
                        "hit_rank": rank, "run_id": "map_v3",
                    })
                    L = qe - qs + 1
                    if fam_s:
                        segs.append({"protein_id": pid, "family_id": fam_s, "family_mode": "strict",
                                     "segment_start_aa": qs, "segment_end_aa": qe,
                                     "best_subfamily_id": subfam, "support_score": row["evalue"],
                                     "support_tool": "mmseqs", "profile_cov": row["profile_cov"],
                                     "segment_len": L})
                    if fam_f:
                        segs.append({"protein_id": pid, "family_id": fam_f, "family_mode": "functional",
                                     "segment_start_aa": qs, "segment_end_aa": qe,
                                     "best_subfamily_id": subfam, "support_score": row["evalue"],
                                     "support_tool": "mmseqs", "profile_cov": row["profile_cov"],
                                     "segment_len": L})
                    rank += 1

    _seg_schema = {
        "protein_id": pl.String, "family_id": pl.String, "family_mode": pl.String,
        "segment_start_aa": pl.Int64, "segment_end_aa": pl.Int64, "best_subfamily_id": pl.String,
        "support_score": pl.Float64, "support_tool": pl.String, "profile_cov": pl.Float64,
        "segment_len": pl.Int64,
    }
    seg_df = pl.from_dicts(segs, schema=_seg_schema) if segs else pl.DataFrame(schema=_seg_schema)

    recs = {r.id: len(r.seq) for r in read_fasta(proteins_fasta)}
    arch = []
    for pid, L in recs.items():
        if len(seg_df) > 0:
            p_segs = seg_df.filter(
                (pl.col("protein_id") == pid) & (pl.col("family_mode") == "strict")
            ).sort("segment_start_aa")
        else:
            p_segs = seg_df
        if len(p_segs) > 0:
            arch_str = "|".join(p_segs["family_id"].to_list())
            cov_aa = sum(p_segs["segment_len"].to_list())
            arch.append({"protein_id": pid, "architecture": arch_str, "n_segments": len(p_segs),
                         "total_covered_aa": cov_aa, "coverage_fraction": min(1.0, cov_aa / L),
                         "is_fusion": 1 if len(p_segs) > 1 else 0})
        else:
            arch.append({"protein_id": pid, "architecture": "", "n_segments": 0,
                         "total_covered_aa": 0, "coverage_fraction": 0.0, "is_fusion": 0})

    _hits_schema = {
        "protein_id": pl.String, "profile_id": pl.String, "subfamily_id": pl.String,
        "family_id": pl.String, "start_aa": pl.Int64, "end_aa": pl.Int64,
        "prob": pl.Float64, "evalue": pl.Float64, "profile_cov": pl.Float64,
        "protein_cov": pl.Float64, "aln_len": pl.Int64, "tool": pl.String,
        "hit_rank": pl.Int64, "run_id": pl.String,
    }
    _arch_schema = {
        "protein_id": pl.String, "architecture": pl.String, "n_segments": pl.Int64,
        "total_covered_aa": pl.Int64, "coverage_fraction": pl.Float64, "is_fusion": pl.Int64,
    }
    hits_df = pl.from_dicts(hits, schema=_hits_schema) if hits else pl.DataFrame(schema=_hits_schema)
    arch_df = pl.from_dicts(arch, schema=_arch_schema) if arch else pl.DataFrame(schema=_arch_schema)

    hits_df.write_csv(out / "protein_vs_profile_hits.tsv", separator="\t")
    seg_df.write_csv(out / "protein_family_segments.tsv", separator="\t")
    arch_df.write_csv(out / "protein_architectures.tsv", separator="\t")


def _write_dense_if_small(sparse: pl.DataFrame, row_col: tuple[str, str], out_path: Path, threshold: int) -> None:
    n = sparse[row_col[0]].n_unique() * sparse[row_col[1]].n_unique()
    if n <= threshold and len(sparse) > 0:
        dense = sparse.pivot(
            values="value", index=row_col[0], on=row_col[1], aggregate_function="first"
        ).fill_null(0)
        dense.write_csv(out_path, separator="\t")


def write_matrices(
    subfamily_map: str, protein_family_segments: str, outdir: str,
    config: dict, logger=None, resume: bool = False
) -> None:
    out = _ensure_dir(outdir)
    if resume and (out / "subfamily_x_protein_sparse.tsv").exists():
        if logger:
            logger.info("Resume: write-matrices outputs already exist in %s, skipping.", str(out))
        return
    smap = pl.read_csv(subfamily_map, separator="\t")
    if Path(protein_family_segments).exists() and Path(protein_family_segments).stat().st_size > 0:
        seg = pl.read_csv(protein_family_segments, separator="\t")
    else:
        seg = pl.DataFrame(schema={"protein_id": pl.String, "family_id": pl.String, "family_mode": pl.String})

    sf = (
        smap.select(["subfamily_id", "protein_id"]).unique()
        .sort(["subfamily_id", "protein_id"])
        .with_columns(pl.lit(1).alias("value"))
    )
    sf.write_csv(out / "subfamily_x_protein_sparse.tsv", separator="\t")

    strict = (
        seg.filter(pl.col("family_mode") == "strict")
        .select(["family_id", "protein_id"]).unique()
        .sort(["family_id", "protein_id"])
        .with_columns(pl.lit(1).alias("value"))
    )
    strict.write_csv(out / "family_strict_x_protein_sparse.tsv", separator="\t")

    func = (
        seg.filter(pl.col("family_mode") == "functional")
        .select(["family_id", "protein_id"]).unique()
        .sort(["family_id", "protein_id"])
        .with_columns(pl.lit(1).alias("value"))
    )
    func.write_csv(out / "family_functional_x_protein_sparse.tsv", separator="\t")

    thr = int(config["outputs"]["write_dense_threshold"])
    _write_dense_if_small(sf, ("subfamily_id", "protein_id"), out / "subfamily_x_protein_dense.tsv", thr)
    _write_dense_if_small(strict, ("family_id", "protein_id"), out / "family_strict_x_protein_dense.tsv", thr)
    _write_dense_if_small(func, ("family_id", "protein_id"), out / "family_functional_x_protein_dense.tsv", thr)

    if config["outputs"].get("write_matrix_market", True):
        from scipy.io import mmwrite
        from scipy.sparse import coo_matrix

        proteins = sorted(sf["protein_id"].unique().to_list())
        pc = {p: i for i, p in enumerate(proteins)}

        def dump(prefix: str, frame: pl.DataFrame, row_name: str):
            rows = sorted(frame[row_name].unique().to_list())
            rc = {r: i for i, r in enumerate(rows)}
            mat = coo_matrix(
                (np.ones(len(frame)),
                 ([rc[r] for r in frame[row_name].to_list()],
                  [pc[p] for p in frame["protein_id"].to_list()])),
                shape=(len(rows), len(proteins)),
            )
            mmwrite(out / f"{prefix}.mtx", mat)
            (out / f"{prefix}_rows.txt").write_text("\n".join(rows) + "\n")
            (out / f"{prefix}_cols.txt").write_text("\n".join(proteins) + "\n")

        dump("subfamily_x_protein", sf, "subfamily_id")
        if len(strict) > 0:
            dump("family_strict_x_protein", strict, "family_id")
        if len(func) > 0:
            dump("family_functional_x_protein", func, "family_id")
