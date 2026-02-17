from __future__ import annotations

import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .io_utils import FastaRecord, read_fasta, write_fasta
from .runtime import require_executables, run_cmd


def _ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


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


def mmseqs_cluster(proteins_fasta: str, outdir: str, config: dict, logger) -> dict[str, str]:
    out = _ensure_dir(outdir)
    mm = config["mmseqs"]
    tools = require_executables(["mmseqs"], config["tools"])
    tmpdir = _ensure_dir(mm["tmpdir"])

    db = out / "proteinsDB"
    clu = out / "clusterDB"
    tsv = out / "clusters.tsv"
    rep_db = out / "repDB"
    run_cmd([tools["mmseqs"], "createdb", proteins_fasta, str(db)], logger)
    run_cmd(
        [
            tools["mmseqs"],
            mm["mode"],
            str(db),
            str(clu),
            str(tmpdir),
            "--min-seq-id",
            str(mm["min_seq_id"]),
            "-c",
            str(mm["coverage"]),
            "--cov-mode",
            str(mm["cov_mode"]),
            "-e",
            str(mm["evalue"]),
            "-s",
            str(mm["sensitivity"]),
            "--threads",
            str(mm["threads"]),
        ],
        logger,
    )
    run_cmd([tools["mmseqs"], "createtsv", str(db), str(db), str(clu), str(tsv)], logger)
    run_cmd([tools["mmseqs"], "result2repseq", str(db), str(clu), str(rep_db)], logger)
    run_cmd([tools["mmseqs"], "result2flat", str(db), str(db), str(rep_db), str(out / "subfamily_reps_raw.faa")], logger)

    raw = pd.read_csv(tsv, sep="\t", names=["rep_protein_id", "protein_id"], header=None)
    reps_sorted = sorted(raw["rep_protein_id"].unique())
    rep_to_sub = {r: f"subfam_{i:06d}" for i, r in enumerate(reps_sorted)}
    raw["subfamily_id"] = raw["rep_protein_id"].map(rep_to_sub)
    raw["is_rep"] = (raw["rep_protein_id"] == raw["protein_id"]).astype(int)
    mapping = raw[["protein_id", "subfamily_id", "is_rep"]].sort_values(["subfamily_id", "protein_id"])
    mapping.to_csv(out / "subfamily_map.tsv", sep="\t", index=False)

    seqs = {r.id: r.seq for r in read_fasta(proteins_fasta)}
    rep_records = [FastaRecord(rep_to_sub[rid], seqs[rid]) for rid in reps_sorted if rid in seqs]
    write_fasta(out / "subfamily_reps.faa", rep_records)

    stats = raw.groupby("subfamily_id")["protein_id"].count().reset_index(name="n_members")
    reps = raw[raw["is_rep"] == 1][["subfamily_id", "protein_id"]].rename(columns={"protein_id": "rep_protein_id"})
    stats = stats.merge(reps, on="subfamily_id", how="left")
    stats["rep_length_aa"] = stats["rep_protein_id"].map(lambda x: len(seqs.get(x, "")))
    stats.sort_values("subfamily_id").to_csv(out / "subfamily_stats.tsv", sep="\t", index=False)
    return tools


def build_profiles(proteins_fasta: str, subfamily_map: str, outdir: str, config: dict, logger) -> dict[str, str]:
    out = _ensure_dir(outdir)
    tools = require_executables(["mafft", "hhmake"], config["tools"])
    smap = pd.read_csv(subfamily_map, sep="\t")
    seqs = {r.id: r.seq for r in read_fasta(proteins_fasta)}
    cap = int(config["profiles"]["max_members_per_subfamily"])

    rows: list[dict] = []
    for subfam, grp in smap.groupby("subfamily_id", sort=True):
        members = sorted(grp["protein_id"].tolist())[:cap]
        fa = out / f"{subfam}.faa"
        a3m = out / f"{subfam}.a3m"
        hhm = out / f"{subfam}.hhm"
        write_fasta(fa, [FastaRecord(pid, seqs[pid]) for pid in members if pid in seqs])
        msa = run_cmd([tools["mafft"], "--auto", str(fa)], logger)
        a3m.write_text(msa)
        run_cmd([tools["hhmake"], "-i", str(a3m), "-o", str(hhm)], logger)
        rows.append(
            {
                "subfamily_id": subfam,
                "hhm_path": str(hhm),
                "msa_path": str(a3m),
                "n_members_used": len(members),
                "build_tool": "mafft+hhmake",
                "build_params_json": json.dumps({"mafft": "--auto", "max_members": cap}),
            }
        )

    pd.DataFrame(rows).sort_values("subfamily_id").to_csv(out / "subfamily_profile_index.tsv", sep="\t", index=False)
    return tools


def hmm_hmm_edges(
    profile_index: str,
    outdir: str,
    config: dict,
    logger,
    candidate_edges_tsv: str | None = None,
) -> dict[str, str]:
    out = _ensure_dir(outdir)
    tools = require_executables(["hhalign"], config["tools"])
    idx = pd.read_csv(profile_index, sep="\t").sort_values("subfamily_id")
    lengths = {r.subfamily_id: _parse_hhm_len(r.hhm_path) for _, r in idx.iterrows()}
    hhm = {r.subfamily_id: r.hhm_path for _, r in idx.iterrows()}

    cands: list[tuple[str, str]] = []
    topn = int(config["hmm_hmm"]["topN"])
    if candidate_edges_tsv and Path(candidate_edges_tsv).exists():
        cdf = pd.read_csv(candidate_edges_tsv, sep="\t")
        for q, g in cdf.groupby("q_subfamily_id"):
            for t in sorted(g.sort_values("cosine", ascending=False)["t_subfamily_id"].head(topn).tolist()):
                cands.append((q, t))
    else:
        subfams = idx["subfamily_id"].tolist()
        for i, q in enumerate(subfams):
            for t in subfams[i + 1 : i + 1 + topn]:
                cands.append((q, t))

    raw_rows: list[dict] = []
    run_id = "hmm_hmm_v2"
    for q, t in sorted(set(tuple(sorted(x)) for x in cands)):
        out_hhr = out / f"{q}__{t}.hhr"
        run_cmd([tools["hhalign"], "-i", hhm[q], "-t", hhm[t], "-o", str(out_hhr)], logger)
        m = _parse_hhr_metrics(out_hhr)
        qlen = max(1, lengths.get(q, 0))
        tlen = max(1, lengths.get(t, 0))
        aln_len = int(m["aln_len"])
        qcov = min(1.0, aln_len / qlen)
        tcov = min(1.0, aln_len / tlen)
        raw_rows.append(
            {
                "q_subfamily_id": q,
                "t_subfamily_id": t,
                "prob": float(m["prob"]),
                "evalue": float(m["evalue"]),
                "bits": "NA",
                "qcov": qcov,
                "tcov": tcov,
                "aln_len": aln_len,
                "pident": float(m["pident"]) if not np.isnan(m["pident"]) else "NA",
                "tool": "hhalign",
                "run_id": run_id,
            }
        )

    raw = pd.DataFrame(raw_rows).sort_values(["q_subfamily_id", "t_subfamily_id"])
    raw.to_csv(out / "hmm_hmm_edges_raw.tsv", sep="\t", index=False)

    raw2 = raw.copy()
    raw2["mincov"] = raw2[["qcov", "tcov"]].min(axis=1)
    raw2["edge_weight"] = (raw2["prob"] / 100.0) * raw2["mincov"]

    hc = config["hmm_hmm"]
    core = raw2[
        (raw2["mincov"] >= float(hc["mincov_core"]))
        & ((raw2["prob"] >= float(hc["min_prob_core"])) | (raw2["evalue"] <= float(hc["max_evalue_core"])))
        & (raw2["aln_len"] >= int(hc["min_aln_len_core"]))
    ].copy()
    core["source"] = "hmm_hmm_core"
    core[
        ["q_subfamily_id", "t_subfamily_id", "edge_weight", "prob", "evalue", "qcov", "tcov", "mincov", "aln_len", "source"]
    ].to_csv(out / "hmm_hmm_edges_core.tsv", sep="\t", index=False)

    relaxed = raw2[
        (raw2["mincov"] >= float(hc["mincov_relaxed"]))
        & ((raw2["prob"] >= float(hc["min_prob_relaxed"])) | (raw2["evalue"] <= float(hc["max_evalue_relaxed"])))
        & (raw2["aln_len"] >= int(hc["min_aln_len_relaxed"]))
    ].copy()
    relaxed["source"] = "hmm_hmm_relaxed"
    relaxed[
        ["q_subfamily_id", "t_subfamily_id", "edge_weight", "prob", "evalue", "qcov", "tcov", "mincov", "aln_len", "source"]
    ].to_csv(out / "hmm_hmm_edges_relaxed.tsv", sep="\t", index=False)
    return tools


def embed(reps_fasta: str, outdir: str, config: dict, weights_path: str | None, logger) -> None:
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
    recs = sorted(read_fasta(reps_fasta), key=lambda x: x.id)
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(weights_path)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    max_len = int(config["embed"]["max_len"])
    policy = config["embed"]["long_seq_policy"]
    seqs = []
    lengths = []
    for r in recs:
        lengths.append(len(r.seq))
        seq = r.seq
        if len(seq) > max_len and policy == "truncate":
            seq = seq[:max_len]
        seqs.append((r.id, seq))

    labels, strs, toks = batch_converter(seqs)
    with torch.no_grad():
        outp = model(toks, repr_layers=[model.num_layers], return_contacts=False)
    reps = outp["representations"][model.num_layers]

    embs = []
    for i, seq in enumerate(strs):
        L = len(seq)
        embs.append(reps[i, 1 : L + 1].mean(0).cpu().numpy())

    mat = np.vstack(embs).astype(np.float32)
    np.save(out / "embeddings.npy", mat)
    (out / "ids.txt").write_text("\n".join([r.id for r in recs]) + "\n")
    pd.DataFrame({"subfamily_id": [r.id for r in recs], "rep_length_aa": lengths}).to_csv(
        out / "lengths.tsv", sep="\t", index=False
    )
    (out / "metadata.json").write_text(
        json.dumps(
            {
                "model_name": config["embed"]["esm_model_name"],
                "weights_path": str(weights_path),
                "pooling": config["embed"]["pooling"],
                "long_seq_policy": policy,
                "max_len": max_len,
                "n_subfamilies": len(recs),
                "dim": int(mat.shape[1]),
            },
            indent=2,
        )
    )


def knn(embeddings_npy: str, ids_txt: str, lengths_tsv: str, out_tsv: str, config: dict) -> None:
    X = np.load(embeddings_npy)
    ids = [x.strip() for x in Path(ids_txt).read_text().splitlines() if x.strip()]
    lens = pd.read_csv(lengths_tsv, sep="\t").set_index("subfamily_id")["rep_length_aa"].to_dict()
    X = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    k = max(1, min(int(config["knn"]["k"]), len(ids) - 1))

    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(X.shape[1])
        index.add(X.astype(np.float32))
        sims, nbrs = index.search(X.astype(np.float32), k + 1)
    except Exception:
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
                rows.append(
                    {
                        "q_subfamily_id": q,
                        "t_subfamily_id": t,
                        "cosine": cosine,
                        "q_len": qlen,
                        "t_len": tlen,
                        "len_ratio": ratio,
                        "pass_len_ratio": pass_lr,
                    }
                )
    pd.DataFrame(rows).sort_values(["q_subfamily_id", "t_subfamily_id"]).to_csv(out_tsv, sep="\t", index=False)


def merge_graph(
    hmm_core_tsv: str,
    emb_tsv: str,
    out_strict_tsv: str,
    out_functional_tsv: str,
    config: dict,
    hmm_relaxed_tsv: str | None = None,
) -> None:
    Path(out_strict_tsv).parent.mkdir(parents=True, exist_ok=True)
    Path(out_functional_tsv).parent.mkdir(parents=True, exist_ok=True)
    core = pd.read_csv(hmm_core_tsv, sep="\t") if Path(hmm_core_tsv).exists() else pd.DataFrame()
    emb = pd.read_csv(emb_tsv, sep="\t") if Path(emb_tsv).exists() else pd.DataFrame()
    relaxed = pd.read_csv(hmm_relaxed_tsv, sep="\t") if hmm_relaxed_tsv and Path(hmm_relaxed_tsv).exists() else pd.DataFrame()

    strict = core.copy()
    strict = strict.rename(columns={"q_subfamily_id": "qid", "t_subfamily_id": "tid"})
    strict["weight"] = strict["edge_weight"]
    strict["source"] = "hmm_core"
    strict[["qid", "tid", "weight", "source"]].sort_values(["qid", "tid"]).to_csv(out_strict_tsv, sep="\t", index=False)

    policy = config["graph"]["edge_weight_policy"]
    edges: dict[tuple[str, str], dict[str, float | str]] = {}

    for df, src in [(core, "hmm_core"), (relaxed, "hmm_relaxed")]:
        for _, r in df.iterrows():
            k = tuple(sorted((r.q_subfamily_id, r.t_subfamily_id)))
            edges[k] = {"hmm": float(r.edge_weight), "emb": 0.0, "source": src}

    for _, r in emb.iterrows():
        k = tuple(sorted((r.q_subfamily_id, r.t_subfamily_id)))
        if policy == "strict":
            continue
        if policy == "gated":
            weak = edges.get(k, {"hmm": 0.0})["hmm"]
            if float(weak) < float(config["graph"]["weak_hmm_threshold"]):
                continue
        if k not in edges:
            edges[k] = {"hmm": 0.0, "emb": float(r.cosine), "source": "emb"}
        else:
            edges[k]["emb"] = max(float(edges[k]["emb"]), float(r.cosine))
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
        rows.append({"qid": q, "tid": t, "weight": weight, "source": str(w["source"]), "hmm_weight": hmm_w, "emb_weight": emb_w})

    pd.DataFrame(rows).sort_values(["qid", "tid"]).to_csv(out_functional_tsv, sep="\t", index=False)


def _cluster_leiden(edges: pd.DataFrame, resolution: float, seed: int) -> pd.DataFrame:
    import igraph as ig
    import leidenalg

    nodes = sorted(set(edges["qid"]).union(set(edges["tid"])))
    n2i = {n: i for i, n in enumerate(nodes)}
    g = ig.Graph(n=len(nodes), edges=[(n2i[a], n2i[b]) for a, b in edges[["qid", "tid"]].itertuples(index=False)], directed=False)
    g.es["weight"] = edges["weight"].tolist()
    part = leidenalg.find_partition(
        g,
        leidenalg.RBConfigurationVertexPartition,
        weights=g.es["weight"],
        resolution_parameter=resolution,
        seed=seed,
    )
    return pd.DataFrame({"subfamily_id": nodes, "community": part.membership})


def cluster_families(
    merged_edges_strict: str,
    merged_edges_functional: str,
    subfamily_map: str,
    outdir: str,
    config: dict,
    method: str = "leiden",
) -> None:
    out = _ensure_dir(outdir)
    if method == "mcl":
        require_executables(["mcl"], config["tools"])
        raise RuntimeError("MCL comparison requested but not implemented in this minimal release.")

    strict_edges = pd.read_csv(merged_edges_strict, sep="\t")
    func_edges = pd.read_csv(merged_edges_functional, sep="\t")

    strict = _cluster_leiden(strict_edges, float(config["graph"]["leiden_resolution_strict"]), int(config["graph"]["seed"]))
    func = _cluster_leiden(func_edges, float(config["graph"]["leiden_resolution_functional"]), int(config["graph"]["seed"]))

    strict["family_id"] = strict["community"].map(lambda x: f"famS_{int(x):06d}")
    func["family_id"] = func["community"].map(lambda x: f"famF_{int(x):06d}")

    strict_out = strict[["subfamily_id", "family_id"]].copy()
    strict_out["method"] = method
    strict_out["method_params_json"] = json.dumps({"resolution": config["graph"]["leiden_resolution_strict"], "seed": config["graph"]["seed"]})
    strict_out["notes"] = "Mode A strict core-domain families"
    strict_out.to_csv(out / "subfamily_to_family_strict.tsv", sep="\t", index=False)
    strict_out.to_csv(out / "subfamily_to_family.tsv", sep="\t", index=False)

    func_out = func[["subfamily_id", "family_id"]].copy()
    func_out["method"] = method
    func_out["method_params_json"] = json.dumps({"resolution": config["graph"]["leiden_resolution_functional"], "seed": config["graph"]["seed"]})
    func_out["notes"] = "Mode B functional neighborhoods"
    func_out.to_csv(out / "subfamily_to_family_functional.tsv", sep="\t", index=False)

    smap = pd.read_csv(subfamily_map, sep="\t")
    for tag, sdf in [("strict", strict_out), ("functional", func_out)]:
        st = sdf.groupby("family_id")["subfamily_id"].count().reset_index(name="n_subfamilies")
        prot = smap.merge(sdf[["subfamily_id", "family_id"]], on="subfamily_id", how="left").groupby("family_id")["protein_id"].nunique().reset_index(name="n_proteins")
        stats = st.merge(prot, on="family_id", how="left")
        stats["edge_density_core"] = np.nan
        stats["mean_rep_len"] = np.nan
        stats["edge_source_counts_json"] = "{}"
        stats.sort_values("family_id").to_csv(out / f"family_stats_{tag}.tsv", sep="\t", index=False)


def map_proteins_to_families(
    proteins_fasta: str,
    subfamily_to_family_strict: str,
    subfamily_to_family_functional: str,
    subfamily_map: str,
    outdir: str,
    config: dict,
) -> None:
    out = _ensure_dir(outdir)
    recs = sorted(read_fasta(proteins_fasta), key=lambda r: r.id)
    smap = pd.read_csv(subfamily_map, sep="\t")
    s2f = pd.read_csv(subfamily_to_family_strict, sep="\t").set_index("subfamily_id")["family_id"].to_dict()
    f2f = pd.read_csv(subfamily_to_family_functional, sep="\t").set_index("subfamily_id")["family_id"].to_dict()
    p2s = smap.groupby("protein_id")["subfamily_id"].first().to_dict()

    hits = []
    segs = []
    arch = []
    for rec in recs:
        pid, L = rec.id, len(rec.seq)
        sub = p2s.get(pid, "")
        fam_s = s2f.get(sub, "")
        fam_f = f2f.get(sub, "")
        if fam_s:
            hits.append({"protein_id": pid, "profile_id": sub, "subfamily_id": sub, "family_id": fam_s, "start_aa": 1, "end_aa": L, "prob": 99.0, "evalue": 1e-20, "profile_cov": 1.0, "protein_cov": 1.0, "aln_len": L, "tool": "containment", "hit_rank": 1, "run_id": "map_v2"})
            segs.append({"protein_id": pid, "family_id": fam_s, "family_mode": "strict", "segment_start_aa": 1, "segment_end_aa": L, "best_subfamily_id": sub, "support_score": 99.0, "support_tool": "containment", "profile_cov": 1.0, "segment_len": L})
            if fam_f:
                segs.append({"protein_id": pid, "family_id": fam_f, "family_mode": "functional", "segment_start_aa": 1, "segment_end_aa": L, "best_subfamily_id": sub, "support_score": 99.0, "support_tool": "containment", "profile_cov": 1.0, "segment_len": L})
            arch.append({"protein_id": pid, "architecture": fam_s, "n_segments": 1, "total_covered_aa": L, "coverage_fraction": 1.0, "is_fusion": 0})
        else:
            arch.append({"protein_id": pid, "architecture": "", "n_segments": 0, "total_covered_aa": 0, "coverage_fraction": 0.0, "is_fusion": 0})

    pd.DataFrame(hits, columns=["protein_id","profile_id","subfamily_id","family_id","start_aa","end_aa","prob","evalue","profile_cov","protein_cov","aln_len","tool","hit_rank","run_id"]).to_csv(out / "protein_vs_profile_hits.tsv", sep="\t", index=False)
    pd.DataFrame(segs, columns=["protein_id","family_id","family_mode","segment_start_aa","segment_end_aa","best_subfamily_id","support_score","support_tool","profile_cov","segment_len"]).to_csv(out / "protein_family_segments.tsv", sep="\t", index=False)
    pd.DataFrame(arch, columns=["protein_id","architecture","n_segments","total_covered_aa","coverage_fraction","is_fusion"]).to_csv(out / "protein_architectures.tsv", sep="\t", index=False)


def _write_dense_if_small(sparse: pd.DataFrame, row_col: tuple[str, str], out_path: Path, threshold: int) -> None:
    n = sparse[row_col[0]].nunique() * sparse[row_col[1]].nunique()
    if n <= threshold and len(sparse):
        sparse.pivot_table(index=row_col[0], columns=row_col[1], values="value", fill_value=0).to_csv(out_path, sep="\t")


def write_matrices(subfamily_map: str, protein_family_segments: str, outdir: str, config: dict) -> None:
    out = _ensure_dir(outdir)
    smap = pd.read_csv(subfamily_map, sep="\t")
    if Path(protein_family_segments).exists() and Path(protein_family_segments).stat().st_size > 0:
        seg = pd.read_csv(protein_family_segments, sep="\t")
    else:
        seg = pd.DataFrame(columns=["protein_id","family_id","family_mode"])

    sf = smap[["subfamily_id", "protein_id"]].drop_duplicates().sort_values(["subfamily_id", "protein_id"])
    sf["value"] = 1
    sf.to_csv(out / "subfamily_x_protein_sparse.tsv", sep="\t", index=False)

    strict = seg[seg["family_mode"] == "strict"][["family_id", "protein_id"]].drop_duplicates().sort_values(["family_id", "protein_id"])
    strict["value"] = 1
    strict.to_csv(out / "family_strict_x_protein_sparse.tsv", sep="\t", index=False)

    func = seg[seg["family_mode"] == "functional"][["family_id", "protein_id"]].drop_duplicates().sort_values(["family_id", "protein_id"])
    func["value"] = 1
    func.to_csv(out / "family_functional_x_protein_sparse.tsv", sep="\t", index=False)

    thr = int(config["outputs"]["write_dense_threshold"])
    _write_dense_if_small(sf, ("subfamily_id", "protein_id"), out / "subfamily_x_protein_dense.tsv", thr)
    _write_dense_if_small(strict, ("family_id", "protein_id"), out / "family_strict_x_protein_dense.tsv", thr)
    _write_dense_if_small(func, ("family_id", "protein_id"), out / "family_functional_x_protein_dense.tsv", thr)

    if config["outputs"].get("write_matrix_market", True):
        from scipy.io import mmwrite
        from scipy.sparse import coo_matrix

        proteins = sorted(sf["protein_id"].unique())
        pc = {p: i for i, p in enumerate(proteins)}

        def dump(prefix: str, frame: pd.DataFrame, row_name: str):
            rows = sorted(frame[row_name].unique())
            rc = {r: i for i, r in enumerate(rows)}
            mat = coo_matrix((np.ones(len(frame)), ([rc[r] for r in frame[row_name]], [pc[p] for p in frame["protein_id"]])), shape=(len(rows), len(proteins)))
            mmwrite(out / f"{prefix}.mtx", mat)
            (out / f"{prefix}_rows.txt").write_text("\n".join(rows) + "\n")
            (out / f"{prefix}_cols.txt").write_text("\n".join(proteins) + "\n")

        dump("subfamily_x_protein", sf, "subfamily_id")
        if len(strict):
            dump("family_strict_x_protein", strict, "family_id")
        if len(func):
            dump("family_functional_x_protein", func, "family_id")
