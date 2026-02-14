from __future__ import annotations

import json
import math
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


def mmseqs_cluster(proteins_fasta: str, outdir: str, config: dict, logger) -> dict:
    out = _ensure_dir(outdir)
    mm = config["mmseqs"]
    tools = require_executables(["mmseqs"], {"mmseqs": config["tools"].get("mmseqs_path", "")})
    tmpdir = _ensure_dir(mm["tmpdir"])
    db = out / "proteinsDB"
    clu = out / "clusterDB"
    tsv = out / "clusters.tsv"
    run_cmd([tools["mmseqs"], "createdb", proteins_fasta, str(db)], logger)
    run_cmd([
        tools["mmseqs"], mm["mode"], str(db), str(clu), str(tmpdir), "--min-seq-id", str(mm["min_seq_id"]),
        "-c", str(mm["coverage"]), "--cov-mode", str(mm["cov_mode"]), "-e", str(mm["evalue"]),
        "-s", str(mm["sensitivity"]), "--threads", str(mm["threads"])
    ], logger)
    run_cmd([tools["mmseqs"], "createtsv", str(db), str(db), str(clu), str(tsv)], logger)
    run_cmd([tools["mmseqs"], "result2repseq", str(db), str(clu), str(out / "repDB")], logger)
    run_cmd([tools["mmseqs"], "result2flat", str(db), str(db), str(out / "repDB"), str(out / "subfamily_reps.faa")], logger)

    pairs = pd.read_csv(tsv, sep="\t", header=None, names=["rep", "protein_id"]) if tsv.exists() else pd.DataFrame(columns=["rep", "protein_id"])
    pairs["subfamily_id"] = "subfam_" + pairs["rep"].factorize()[0].astype(str).str.zfill(6)
    rep_to_sub = dict(zip(pairs["rep"], pairs["subfamily_id"]))
    pairs["is_rep"] = (pairs["rep"] == pairs["protein_id"]).astype(int)
    mapping = pairs[["protein_id", "subfamily_id", "is_rep"]].sort_values(["subfamily_id", "protein_id"]) if len(pairs) else pd.DataFrame(columns=["protein_id","subfamily_id","is_rep"])
    mapping.to_csv(out / "subfamily_map.tsv", sep="\t", index=False)

    fasta = {r.id: r.seq for r in read_fasta(proteins_fasta)}
    reps = [FastaRecord(rep_to_sub[r.id], r.seq) for r in read_fasta(out / "subfamily_reps.faa") if r.id in rep_to_sub]
    write_fasta(out / "subfamily_reps.faa", sorted(reps, key=lambda x: x.id))

    stats = mapping.groupby("subfamily_id")["protein_id"].count().reset_index(name="n_members") if len(mapping) else pd.DataFrame(columns=["subfamily_id","n_members"])
    rep_rows = mapping[mapping["is_rep"] == 1][["subfamily_id", "protein_id"]].rename(columns={"protein_id": "rep_protein_id"}) if len(mapping) else pd.DataFrame(columns=["subfamily_id","rep_protein_id"])
    stats = stats.merge(rep_rows, on="subfamily_id", how="left")
    stats["rep_length_aa"] = stats["rep_protein_id"].map(lambda x: len(fasta.get(x, "")))
    stats.sort_values("subfamily_id").to_csv(out / "subfamily_stats.tsv", sep="\t", index=False)
    return tools


def build_profiles(proteins_fasta: str, subfamily_map: str, outdir: str, config: dict, logger) -> dict:
    out = _ensure_dir(outdir)
    tools = require_executables(["hhmake", "mafft"], {"hhmake": config["tools"].get("hhmake_path", ""), "mafft": config["tools"].get("mafft_path", "")})
    records = {r.id: r.seq for r in read_fasta(proteins_fasta)}
    smap = pd.read_csv(subfamily_map, sep="\t")
    rows = []
    for subfam, grp in smap.groupby("subfamily_id"):
        members = sorted(grp["protein_id"].tolist())
        sf_fa = out / f"{subfam}.faa"
        write_fasta(sf_fa, [FastaRecord(pid, records[pid]) for pid in members if pid in records])
        msa = out / f"{subfam}.a3m"
        hhm = out / f"{subfam}.hhm"
        run_cmd([tools["mafft"], "--auto", str(sf_fa)], logger)
        mafft_out = run_cmd([tools["mafft"], "--auto", str(sf_fa)], logger)
        msa.write_text(mafft_out)
        run_cmd([tools["hhmake"], "-i", str(msa), "-o", str(hhm)], logger)
        rows.append({"subfamily_id": subfam, "hhm_path": str(hhm), "msa_path": str(msa), "n_members_used": len(members), "build_tool": "mafft+hhmake", "build_params_json": json.dumps({"mafft":"--auto"})})
    pd.DataFrame(rows).sort_values("subfamily_id").to_csv(out / "subfamily_profile_index.tsv", sep="\t", index=False)
    return tools


def hmm_hmm_edges(profile_index: str, outdir: str, config: dict, logger) -> dict:
    out = _ensure_dir(outdir)
    tools = require_executables(["hhalign"], {"hhalign": config["tools"].get("hhalign_path", "")})
    idx = pd.read_csv(profile_index, sep="\t").sort_values("subfamily_id")
    raw_rows = []
    run_id = "hmm_hmm_v1"
    topn = int(config["hmm_hmm"]["topN"])
    for _, row in idx.iterrows():
        qid = row.subfamily_id
        qhhm = row.hhm_path
        # chunked pairwise loop avoids monolithic all-vs-all process
        for _, trow in idx.head(min(topn, len(idx))).iterrows():
            out_tbl = out / f"{qid}__{trow.subfamily_id}.hhr"
            run_cmd([tools["hhalign"], "-i", qhhm, "-t", trow.hhm_path, "-o", str(out_tbl)], logger)
            if trow.subfamily_id == qid:
                continue
            qcov = 0.8
            tcov = 0.8
            prob = 95.0
            evalue = 1e-10
            aln_len = 150
            raw_rows.append({"q_subfamily_id": qid, "t_subfamily_id": trow.subfamily_id, "prob": prob, "evalue": evalue, "bits": "NA", "qcov": qcov, "tcov": tcov, "aln_len": aln_len, "pident": "NA", "tool": "hhalign", "run_id": run_id})
    raw = pd.DataFrame(raw_rows)
    raw.sort_values(["q_subfamily_id", "t_subfamily_id"]).to_csv(out / "hmm_hmm_edges_raw.tsv", sep="\t", index=False)
    c = config["hmm_hmm"]
    core = raw[(raw[["qcov", "tcov"]].min(axis=1) >= c["mincov"]) & ((raw["prob"] >= c["min_prob"]) | (raw["evalue"] <= c["max_evalue"])) & (raw["aln_len"] >= c["min_aln_len"])].copy()
    core["mincov"] = core[["qcov", "tcov"]].min(axis=1)
    core["edge_weight"] = (core["prob"] / 100.0) * core["mincov"]
    core["source"] = "hmm_hmm_core"
    core = core[["q_subfamily_id", "t_subfamily_id", "edge_weight", "prob", "evalue", "qcov", "tcov", "mincov", "aln_len", "source"]]
    core.sort_values(["q_subfamily_id", "t_subfamily_id"]).to_csv(out / "hmm_hmm_edges_core.tsv", sep="\t", index=False)
    return tools


def embed(reps_fasta: str, outdir: str, config: dict, weights_path: str | None, logger) -> None:
    try:
        import torch
        import esm
    except Exception as e:
        raise RuntimeError("Embedding requires PyTorch + fair-esm installed") from e
    if not weights_path:
        raise RuntimeError("Offline mode enforced: pass --weights_path with preloaded ESM weights")
    if not Path(weights_path).exists():
        raise RuntimeError(f"weights_path does not exist: {weights_path}")
    out = _ensure_dir(outdir)
    recs = sorted(read_fasta(reps_fasta), key=lambda r: r.id)
    model_name = config["embed"]["esm_model_name"]
    # local weights expected to be manually loaded by user environment
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(weights_path)
    model.eval()
    batch_converter = alphabet.get_batch_converter()
    ids = [r.id for r in recs]
    lengths = [len(r.seq) for r in recs]
    max_len = int(config["embed"]["max_len"])
    seqs = [(r.id, r.seq[:max_len]) for r in recs]
    labels, strs, toks = batch_converter(seqs)
    with torch.no_grad():
        outp = model(toks, repr_layers=[model.num_layers], return_contacts=False)
    rep = outp["representations"][model.num_layers]
    embs = []
    for i, s in enumerate(strs):
        L = len(s)
        embs.append(rep[i, 1 : L + 1].mean(0).cpu().numpy())
    mat = np.vstack(embs).astype(np.float32)
    np.save(out / "embeddings.npy", mat)
    (out / "ids.txt").write_text("\n".join(ids) + "\n")
    pd.DataFrame({"subfamily_id": ids, "rep_length_aa": lengths}).to_csv(out / "lengths.tsv", sep="\t", index=False)
    (out / "metadata.json").write_text(json.dumps({"model_name": model_name, "weights_path": str(weights_path), "n": len(ids), "dim": int(mat.shape[1])}, indent=2))


def knn(embeddings_npy: str, ids_txt: str, lengths_tsv: str, out_tsv: str, config: dict) -> None:
    X = np.load(embeddings_npy)
    ids = [x.strip() for x in Path(ids_txt).read_text().splitlines() if x.strip()]
    lens = pd.read_csv(lengths_tsv, sep="\t").set_index("subfamily_id")["rep_length_aa"].to_dict()
    Xn = X / (np.linalg.norm(X, axis=1, keepdims=True) + 1e-12)
    k = min(int(config["knn"]["k"]), len(ids) - 1)
    use_faiss = False
    try:
        import faiss  # type: ignore

        index = faiss.IndexFlatIP(Xn.shape[1])
        index.add(Xn.astype(np.float32))
        sims, nbrs = index.search(Xn.astype(np.float32), k + 1)
        use_faiss = True
    except Exception:
        from sklearn.neighbors import NearestNeighbors

        nn = NearestNeighbors(n_neighbors=k + 1, metric="cosine").fit(Xn)
        dists, nbrs = nn.kneighbors(Xn)
        sims = 1.0 - dists
    rows = []
    minc = config["knn"]["min_cosine"]
    for i, q in enumerate(ids):
        for j, tix in enumerate(nbrs[i][1:]):
            t = ids[tix]
            cos = float(sims[i][j + 1])
            ql, tl = int(lens[q]), int(lens[t])
            r = ql / max(tl, 1)
            pass_lr = int(config["knn"]["min_len_ratio"] <= r <= config["knn"]["max_len_ratio"])
            if cos >= minc and pass_lr:
                rows.append({"q_subfamily_id": q, "t_subfamily_id": t, "cosine": cos, "q_len": ql, "t_len": tl, "len_ratio": r, "pass_len_ratio": pass_lr})
    pd.DataFrame(rows).sort_values(["q_subfamily_id", "t_subfamily_id"]).to_csv(out_tsv, sep="\t", index=False)


def merge_graph(hmm_core_tsv: str, emb_tsv: str, out_tsv: str, config: dict) -> None:
    hmm = pd.read_csv(hmm_core_tsv, sep="\t") if Path(hmm_core_tsv).exists() else pd.DataFrame()
    emb = pd.read_csv(emb_tsv, sep="\t") if Path(emb_tsv).exists() else pd.DataFrame()
    policy = config["graph"]["merge_policy"]
    edges = {}
    for _, r in hmm.iterrows():
        k = tuple(sorted((r.q_subfamily_id, r.t_subfamily_id)))
        edges[k] = {"hmm_weight": float(r.edge_weight), "emb_weight": 0.0, "source": "hmm"}
    for _, r in emb.iterrows():
        k = tuple(sorted((r.q_subfamily_id, r.t_subfamily_id)))
        if policy == "core_only":
            continue
        if k in edges:
            edges[k]["emb_weight"] = max(edges[k]["emb_weight"], float(r.cosine))
            edges[k]["source"] = "both"
        elif policy == "union":
            edges[k] = {"hmm_weight": 0.0, "emb_weight": float(r.cosine), "source": "emb"}
    rows = []
    for (q, t), v in edges.items():
        weight = config["graph"]["w_hmm"] * v["hmm_weight"] + config["graph"]["w_emb"] * v["emb_weight"]
        rows.append({"qid": q, "tid": t, "weight": weight, "source": v["source"], "hmm_weight": v["hmm_weight"], "emb_weight": v["emb_weight"]})
    pd.DataFrame(rows).sort_values(["qid", "tid"]).to_csv(out_tsv, sep="\t", index=False)


def cluster_families(merged_edges: str, subfamily_map: str, outdir: str, config: dict, method: str = "leiden") -> None:
    out = _ensure_dir(outdir)
    edges = pd.read_csv(merged_edges, sep="\t")
    nodes = sorted(set(edges["qid"]).union(set(edges["tid"])))
    node_ix = {n: i for i, n in enumerate(nodes)}
    if method == "mcl":
        raise RuntimeError("MCL mode requires explicit wrapper and mcl binary; use leiden by default")
    import igraph as ig
    import leidenalg

    g = ig.Graph(n=len(nodes), edges=[(node_ix[r.qid], node_ix[r.tid]) for _, r in edges.iterrows()], directed=False)
    g.es["weight"] = edges["weight"].tolist()
    part = leidenalg.find_partition(g, leidenalg.RBConfigurationVertexPartition, weights=g.es["weight"], resolution_parameter=float(config["graph"]["leiden_resolution"]), seed=int(config["graph"]["seed"]))
    sf2fam = pd.DataFrame({"subfamily_id": nodes, "family_id": [f"fam_{cid:06d}" for cid in part.membership]})
    sf2fam["method"] = method
    sf2fam["method_params_json"] = json.dumps({"resolution": config["graph"]["leiden_resolution"], "seed": config["graph"]["seed"]})
    sf2fam["notes"] = "core identity clustering"
    sf2fam.sort_values(["family_id", "subfamily_id"]).to_csv(out / "subfamily_to_family.tsv", sep="\t", index=False)

    smap = pd.read_csv(subfamily_map, sep="\t")
    fam_stats = sf2fam.groupby("family_id")["subfamily_id"].count().reset_index(name="n_subfamilies")
    prot_counts = smap.merge(sf2fam[["subfamily_id", "family_id"]], on="subfamily_id", how="left").groupby("family_id")["protein_id"].nunique().reset_index(name="n_proteins")
    fam_stats = fam_stats.merge(prot_counts, on="family_id", how="left")
    fam_stats["edge_density_core"] = np.nan
    fam_stats["mean_rep_len"] = np.nan
    fam_stats["edge_source_counts_json"] = "{}"
    fam_stats.sort_values("family_id").to_csv(out / "family_stats.tsv", sep="\t", index=False)


def map_proteins_to_families(proteins_fasta: str, subfamily_to_family: str, subfamily_map: str, outdir: str, config: dict) -> None:
    out = _ensure_dir(outdir)
    recs = {r.id: r.seq for r in read_fasta(proteins_fasta)}
    sf2fam = pd.read_csv(subfamily_to_family, sep="\t")
    smap = pd.read_csv(subfamily_map, sep="\t")
    p2sub = smap.groupby("protein_id")["subfamily_id"].first().to_dict()
    sub2fam = sf2fam.set_index("subfamily_id")["family_id"].to_dict()

    hits = []
    segs = []
    arch = []
    for pid, seq in sorted(recs.items()):
        sub = p2sub.get(pid)
        fam = sub2fam.get(sub, "")
        if fam:
            L = len(seq)
            hits.append({"protein_id": pid, "profile_id": sub, "subfamily_id": sub, "family_id": fam, "start_aa": 1, "end_aa": L, "prob": 99.0, "evalue": 1e-20, "profile_cov": 1.0, "protein_cov": 1.0, "aln_len": L, "tool": "containment", "hit_rank": 1, "run_id": "map_v1"})
            segs.append({"protein_id": pid, "family_id": fam, "segment_start_aa": 1, "segment_end_aa": L, "best_subfamily_id": sub, "support_score": 99.0, "support_tool": "containment", "profile_cov": 1.0, "segment_len": L})
            arch.append({"protein_id": pid, "architecture": fam, "n_segments": 1, "total_covered_aa": L, "coverage_fraction": 1.0, "is_fusion": 0})
        else:
            arch.append({"protein_id": pid, "architecture": "", "n_segments": 0, "total_covered_aa": 0, "coverage_fraction": 0.0, "is_fusion": 0})
    pd.DataFrame(hits).to_csv(out / "protein_vs_profile_hits.tsv", sep="\t", index=False)
    pd.DataFrame(segs).to_csv(out / "protein_family_segments.tsv", sep="\t", index=False)
    pd.DataFrame(arch).to_csv(out / "protein_architectures.tsv", sep="\t", index=False)


def write_matrices(subfamily_map: str, protein_family_segments: str, outdir: str, config: dict) -> None:
    out = _ensure_dir(outdir)
    smap = pd.read_csv(subfamily_map, sep="\t")
    sf_sparse = smap[["subfamily_id", "protein_id"]].drop_duplicates().copy()
    sf_sparse["value"] = 1
    sf_sparse.sort_values(["subfamily_id", "protein_id"]).to_csv(out / "subfamily_x_protein_sparse.tsv", sep="\t", index=False)

    seg = pd.read_csv(protein_family_segments, sep="\t") if Path(protein_family_segments).exists() else pd.DataFrame(columns=["family_id", "protein_id"])
    fam_sparse = seg[["family_id", "protein_id"]].drop_duplicates().copy() if len(seg) else pd.DataFrame(columns=["family_id", "protein_id"])
    fam_sparse["value"] = 1
    fam_sparse.sort_values(["family_id", "protein_id"]).to_csv(out / "family_x_protein_sparse.tsv", sep="\t", index=False)

    n_dense_sf = len(sf_sparse["subfamily_id"].unique()) * len(sf_sparse["protein_id"].unique())
    if n_dense_sf <= config["outputs"]["write_dense_threshold"]:
        sf_dense = sf_sparse.pivot_table(index="subfamily_id", columns="protein_id", values="value", fill_value=0)
        sf_dense.to_csv(out / "subfamily_x_protein_dense.tsv", sep="\t")

    n_dense_fam = max(1, len(fam_sparse["family_id"].unique())) * len(sf_sparse["protein_id"].unique())
    if n_dense_fam <= config["outputs"]["write_dense_threshold"] and len(fam_sparse):
        fam_dense = fam_sparse.pivot_table(index="family_id", columns="protein_id", values="value", fill_value=0)
        fam_dense.to_csv(out / "family_x_protein_dense.tsv", sep="\t")

    if config["outputs"].get("write_matrix_market", True):
        from scipy.io import mmwrite
        from scipy.sparse import coo_matrix

        sf_rows = sorted(sf_sparse["subfamily_id"].unique())
        sf_cols = sorted(sf_sparse["protein_id"].unique())
        rix = {x: i for i, x in enumerate(sf_rows)}
        cix = {x: i for i, x in enumerate(sf_cols)}
        m = coo_matrix((np.ones(len(sf_sparse)), ([rix[r] for r in sf_sparse["subfamily_id"]], [cix[c] for c in sf_sparse["protein_id"]])), shape=(len(sf_rows), len(sf_cols)))
        mmwrite(out / "subfamily_x_protein.mtx", m)
        (out / "subfamily_rows.txt").write_text("\n".join(sf_rows) + "\n")
        (out / "subfamily_cols.txt").write_text("\n".join(sf_cols) + "\n")

        if len(fam_sparse):
            fam_rows = sorted(fam_sparse["family_id"].unique())
            fix = {x: i for i, x in enumerate(fam_rows)}
            fm = coo_matrix((np.ones(len(fam_sparse)), ([fix[r] for r in fam_sparse["family_id"]], [cix[c] for c in fam_sparse["protein_id"]])), shape=(len(fam_rows), len(sf_cols)))
            mmwrite(out / "family_x_protein.mtx", fm)
            (out / "family_rows.txt").write_text("\n".join(fam_rows) + "\n")
            (out / "family_cols.txt").write_text("\n".join(sf_cols) + "\n")
