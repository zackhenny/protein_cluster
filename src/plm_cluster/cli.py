from __future__ import annotations

import argparse
from pathlib import Path

from .config import load_config
from .pipeline import (
    build_profiles,
    cluster_families,
    embed,
    hmm_hmm_edges,
    knn,
    map_proteins_to_families,
    merge_graph,
    mmseqs_cluster,
    write_matrices,
)
from .runtime import setup_logging, write_manifest


def add_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", default=None)
    parser.add_argument("--results_root", default="results")


def main() -> None:
    ap = argparse.ArgumentParser(prog="plm_cluster")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("mmseqs-cluster")
    add_common(p)
    p.add_argument("--proteins_fasta", required=True)
    p.add_argument("--outdir", default="results/01_mmseqs")

    p = sub.add_parser("build-profiles")
    add_common(p)
    p.add_argument("--proteins_fasta", required=True)
    p.add_argument("--subfamily_map", required=True)
    p.add_argument("--outdir", default="results/02_profiles")

    p = sub.add_parser("hmm-hmm-edges")
    add_common(p)
    p.add_argument("--profile_index", required=True)
    p.add_argument("--outdir", default="results/03_hmm_hmm_edges")

    p = sub.add_parser("embed")
    add_common(p)
    p.add_argument("--reps_fasta", required=True)
    p.add_argument("--weights_path", default=None)
    p.add_argument("--outdir", default="results/04_embeddings")

    p = sub.add_parser("knn")
    add_common(p)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--lengths", required=True)
    p.add_argument("--out_tsv", default="results/04_embeddings/embedding_knn_edges.tsv")

    p = sub.add_parser("merge-graph")
    add_common(p)
    p.add_argument("--hmm_core", required=True)
    p.add_argument("--embedding_edges", required=True)
    p.add_argument("--out_tsv", default="results/06_family_clustering/merged_edges.tsv")

    p = sub.add_parser("cluster-families")
    add_common(p)
    p.add_argument("--merged_edges", required=True)
    p.add_argument("--subfamily_map", required=True)
    p.add_argument("--outdir", default="results/06_family_clustering")
    p.add_argument("--method", default="leiden", choices=["leiden", "mcl"])

    p = sub.add_parser("map-proteins-to-families")
    add_common(p)
    p.add_argument("--proteins_fasta", required=True)
    p.add_argument("--subfamily_to_family", required=True)
    p.add_argument("--subfamily_map", required=True)
    p.add_argument("--outdir", default="results/05_domain_hits")

    p = sub.add_parser("write-matrices")
    add_common(p)
    p.add_argument("--subfamily_map", required=True)
    p.add_argument("--protein_family_segments", required=True)
    p.add_argument("--outdir", default="results/07_membership_matrices")

    p = sub.add_parser("run-all")
    add_common(p)
    p.add_argument("--proteins_fasta", required=True)
    p.add_argument("--weights_path", default=None)

    args = ap.parse_args()
    cfg = load_config(args.config)
    logs = Path(args.results_root) / "logs"
    logger = setup_logging(logs, args.cmd)

    tool_paths = {}
    if args.cmd == "mmseqs-cluster":
        tool_paths = mmseqs_cluster(args.proteins_fasta, args.outdir, cfg, logger)
    elif args.cmd == "build-profiles":
        tool_paths = build_profiles(args.proteins_fasta, args.subfamily_map, args.outdir, cfg, logger)
    elif args.cmd == "hmm-hmm-edges":
        tool_paths = hmm_hmm_edges(args.profile_index, args.outdir, cfg, logger)
    elif args.cmd == "embed":
        embed(args.reps_fasta, args.outdir, cfg, args.weights_path, logger)
    elif args.cmd == "knn":
        knn(args.embeddings, args.ids, args.lengths, args.out_tsv, cfg)
    elif args.cmd == "merge-graph":
        merge_graph(args.hmm_core, args.embedding_edges, args.out_tsv, cfg)
    elif args.cmd == "cluster-families":
        cluster_families(args.merged_edges, args.subfamily_map, args.outdir, cfg, args.method)
    elif args.cmd == "map-proteins-to-families":
        map_proteins_to_families(args.proteins_fasta, args.subfamily_to_family, args.subfamily_map, args.outdir, cfg)
    elif args.cmd == "write-matrices":
        write_matrices(args.subfamily_map, args.protein_family_segments, args.outdir, cfg)
    elif args.cmd == "run-all":
        root = Path(args.results_root)
        mmseqs_cluster(args.proteins_fasta, root / "01_mmseqs", cfg, logger)
        build_profiles(args.proteins_fasta, root / "01_mmseqs/subfamily_map.tsv", root / "02_profiles", cfg, logger)
        hmm_hmm_edges(root / "02_profiles/subfamily_profile_index.tsv", root / "03_hmm_hmm_edges", cfg, logger)
        embed(root / "01_mmseqs/subfamily_reps.faa", root / "04_embeddings", cfg, args.weights_path, logger)
        knn(root / "04_embeddings/embeddings.npy", root / "04_embeddings/ids.txt", root / "04_embeddings/lengths.tsv", root / "04_embeddings/embedding_knn_edges.tsv", cfg)
        merge_graph(root / "03_hmm_hmm_edges/hmm_hmm_edges_core.tsv", root / "04_embeddings/embedding_knn_edges.tsv", root / "06_family_clustering/merged_edges.tsv", cfg)
        cluster_families(root / "06_family_clustering/merged_edges.tsv", root / "01_mmseqs/subfamily_map.tsv", root / "06_family_clustering", cfg)
        map_proteins_to_families(args.proteins_fasta, root / "06_family_clustering/subfamily_to_family.tsv", root / "01_mmseqs/subfamily_map.tsv", root / "05_domain_hits", cfg)
        write_matrices(root / "01_mmseqs/subfamily_map.tsv", root / "05_domain_hits/protein_family_segments.tsv", root / "07_membership_matrices", cfg)

    write_manifest(Path(args.results_root) / "manifests" / f"{args.cmd}_manifest.json", vars(args), tool_paths, [getattr(args, "proteins_fasta", "")])


if __name__ == "__main__":
    main()
