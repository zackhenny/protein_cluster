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
from .runtime import executable_version, require_executables, setup_logging, write_manifest


ALIASES = {
    "map-proteins": "map-proteins-to-families",
    "cluster": "cluster-families",
}


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None)
    p.add_argument("--results_root", default="results")


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

    p = sub.add_parser("embed")
    add_common(p)
    p.add_argument("--reps_fasta", required=True)
    p.add_argument("--weights_path", required=True)
    p.add_argument("--outdir", default="results/04_embeddings")

    p = sub.add_parser("knn")
    add_common(p)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--lengths", required=True)
    p.add_argument("--out_tsv", default="results/04_embeddings/embedding_knn_edges.tsv")

    p = sub.add_parser("hmm-hmm-edges")
    add_common(p)
    p.add_argument("--profile_index", required=True)
    p.add_argument("--candidate_edges", default=None)
    p.add_argument("--outdir", default="results/03_hmm_hmm_edges")

    p = sub.add_parser("merge-graph")
    add_common(p)
    p.add_argument("--hmm_core", required=True)
    p.add_argument("--embedding_edges", required=True)
    p.add_argument("--hmm_relaxed", default=None)
    p.add_argument("--outdir", default="results/06_family_clustering")

    p = sub.add_parser("cluster-families")
    add_common(p)
    p.add_argument("--merged_edges_strict", required=True)
    p.add_argument("--merged_edges_functional", required=True)
    p.add_argument("--subfamily_map", required=True)
    p.add_argument("--outdir", default="results/06_family_clustering")
    p.add_argument("--method", default="leiden", choices=["leiden", "mcl"])

    p = sub.add_parser("map-proteins-to-families")
    add_common(p)
    p.add_argument("--proteins_fasta", required=True)
    p.add_argument("--subfamily_to_family_strict", required=True)
    p.add_argument("--subfamily_to_family_functional", required=True)
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
    p.add_argument("--weights_path", required=True)

    args = ap.parse_args()
    cmd = ALIASES.get(args.cmd, args.cmd)
    cfg = load_config(args.config)
    logger = setup_logging(Path(args.results_root) / "logs", cmd)

    manifest_tools: dict[str, str] = {}
    try:
        manifest_tools.update(require_executables(["mmseqs"], cfg["tools"]))
        manifest_tools.update(require_executables(["hhmake", "hhalign"], cfg["tools"]))
    except Exception as exc:
        logger.info("Runtime tool check deferred: %s", exc)
    else:
        logger.info("mmseqs version: %s", executable_version(manifest_tools["mmseqs"]))
        logger.info("hhalign version: %s", executable_version(manifest_tools["hhalign"]))
        logger.info("hhmake version: %s", executable_version(manifest_tools["hhmake"]))

    if cmd == "mmseqs-cluster":
        manifest_tools.update(mmseqs_cluster(args.proteins_fasta, args.outdir, cfg, logger))
    elif cmd == "build-profiles":
        manifest_tools.update(build_profiles(args.proteins_fasta, args.subfamily_map, args.outdir, cfg, logger))
    elif cmd == "embed":
        embed(args.reps_fasta, args.outdir, cfg, args.weights_path, logger)
    elif cmd == "knn":
        knn(args.embeddings, args.ids, args.lengths, args.out_tsv, cfg)
    elif cmd == "hmm-hmm-edges":
        manifest_tools.update(hmm_hmm_edges(args.profile_index, args.outdir, cfg, logger, args.candidate_edges))
    elif cmd == "merge-graph":
        out = Path(args.outdir)
        out.mkdir(parents=True, exist_ok=True)
        merge_graph(
            args.hmm_core,
            args.embedding_edges,
            str(out / "merged_edges_strict.tsv"),
            str(out / "merged_edges_functional.tsv"),
            cfg,
            args.hmm_relaxed,
        )
    elif cmd == "cluster-families":
        cluster_families(args.merged_edges_strict, args.merged_edges_functional, args.subfamily_map, args.outdir, cfg, args.method)
    elif cmd == "map-proteins-to-families":
        map_proteins_to_families(
            args.proteins_fasta,
            args.subfamily_to_family_strict,
            args.subfamily_to_family_functional,
            args.subfamily_map,
            args.outdir,
            cfg,
        )
    elif cmd == "write-matrices":
        write_matrices(args.subfamily_map, args.protein_family_segments, args.outdir, cfg)
    elif cmd == "run-all":
        root = Path(args.results_root)
        logger.info("Starting run-all pipeline")
        
        logger.info("Step 1/8: MMseqs clustering")
        mmseqs_cluster(args.proteins_fasta, str(root / "01_mmseqs"), cfg, logger)
        
        logger.info("Step 2/8: Building profiles")
        build_profiles(args.proteins_fasta, str(root / "01_mmseqs/subfamily_map.tsv"), str(root / "02_profiles"), cfg, logger)
        
        logger.info("Step 3/8: Embedding subfamily representatives")
        embed(str(root / "01_mmseqs/subfamily_reps.faa"), str(root / "04_embeddings"), cfg, args.weights_path, logger)
        
        logger.info("Step 4/8: Computing KNN edges")
        knn(str(root / "04_embeddings/embeddings.npy"), str(root / "04_embeddings/ids.txt"), str(root / "04_embeddings/lengths.tsv"), str(root / "04_embeddings/embedding_knn_edges.tsv"), cfg)
        
        logger.info("Step 5/8: Computing HMM-HMM edges")
        hmm_hmm_edges(str(root / "02_profiles/subfamily_profile_index.tsv"), str(root / "03_hmm_hmm_edges"), cfg, logger, str(root / "04_embeddings/embedding_knn_edges.tsv"))
        
        logger.info("Step 6/8: Merging graphs")
        merge_graph(
            str(root / "03_hmm_hmm_edges/hmm_hmm_edges_core.tsv"),
            str(root / "04_embeddings/embedding_knn_edges.tsv"),
            str(root / "06_family_clustering/merged_edges_strict.tsv"),
            str(root / "06_family_clustering/merged_edges_functional.tsv"),
            cfg,
            str(root / "03_hmm_hmm_edges/hmm_hmm_edges_relaxed.tsv"),
        )
        
        logger.info("Step 7/8: Clustering families")
        cluster_families(
            str(root / "06_family_clustering/merged_edges_strict.tsv"),
            str(root / "06_family_clustering/merged_edges_functional.tsv"),
            str(root / "01_mmseqs/subfamily_map.tsv"),
            str(root / "06_family_clustering"),
            cfg,
            "leiden",
        )
        
        logger.info("Step 8/8: Mapping proteins to families and writing matrices")
        map_proteins_to_families(
            args.proteins_fasta,
            str(root / "06_family_clustering/subfamily_to_family_strict.tsv"),
            str(root / "06_family_clustering/subfamily_to_family_functional.tsv"),
            str(root / "01_mmseqs/subfamily_map.tsv"),
            str(root / "05_domain_hits"),
            cfg,
        )
        write_matrices(str(root / "01_mmseqs/subfamily_map.tsv"), str(root / "05_domain_hits/protein_family_segments.tsv"), str(root / "07_membership_matrices"), cfg)
        
        logger.info("Pipeline completed successfully")

    write_manifest(
        Path(args.results_root) / "manifests" / "run_manifest.json",
        {"command": cmd, **vars(args)},
        manifest_tools,
        [getattr(args, "proteins_fasta", "")],
    )


if __name__ == "__main__":
    main()
