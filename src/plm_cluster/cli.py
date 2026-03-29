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
    merge_hmm_shards,
    mmseqs_cluster,
    write_matrices,
)
from .qc_plots import generate_qc_plots
from .runtime import add_step_log_handler, executable_version, require_executables, setup_logging, write_manifest


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
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if its output files already exist")

    p = sub.add_parser("build-profiles")
    add_common(p)
    p.add_argument("--proteins_fasta", required=True)
    p.add_argument("--subfamily_map", required=True)
    p.add_argument("--outdir", default="results/02_profiles")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-built per-subfamily profiles; rebuild only missing ones")

    p = sub.add_parser("embed")
    add_common(p)
    p.add_argument("--reps_fasta", required=True)
    p.add_argument("--weights_path", required=True)
    p.add_argument("--outdir", default="results/04_embeddings")
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if embeddings.npy already exists")

    p = sub.add_parser("knn")
    add_common(p)
    p.add_argument("--embeddings", required=True)
    p.add_argument("--ids", required=True)
    p.add_argument("--lengths", required=True)
    p.add_argument("--out_tsv", default="results/04_embeddings/embedding_knn_edges.tsv")
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if the output TSV already exists")

    p = sub.add_parser("hmm-hmm-edges")
    add_common(p)
    p.add_argument("--profile_index", required=True)
    p.add_argument("--candidate_edges", default=None)
    p.add_argument("--mode", default=None, choices=["pairwise", "db-search"],
                   help="Execution mode: 'pairwise' (default) or 'db-search' (hhsearch DB search)")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-completed pairs using the NDJSON progress log")
    p.add_argument("--shard-id", type=int, default=0, dest="shard_id",
                   help="Zero-based shard index for parallel execution (default: 0)")
    p.add_argument("--n-shards", type=int, default=1, dest="n_shards",
                   help="Total number of shards (default: 1 = no sharding)")
    p.add_argument("--outdir", default="results/03_hmm_hmm_edges")

    p = sub.add_parser("merge-hmm-shards")
    add_common(p)
    p.add_argument("--outdir", default="results/03_hmm_hmm_edges",
                   help="Directory containing per-shard TSV files to merge")
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if the merged output already exists")

    p = sub.add_parser("merge-graph")
    add_common(p)
    p.add_argument("--hmm_core", required=True)
    p.add_argument("--embedding_edges", required=True)
    p.add_argument("--hmm_relaxed", default=None)
    p.add_argument("--outdir", default="results/06_family_clustering")
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if merged graph files already exist")

    p = sub.add_parser("cluster-families")
    add_common(p)
    p.add_argument("--merged_edges_strict", required=True)
    p.add_argument("--merged_edges_functional", required=True)
    p.add_argument("--subfamily_map", required=True)
    p.add_argument("--outdir", default="results/06_family_clustering")
    p.add_argument("--method", default="leiden", choices=["leiden", "mcl"])
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if family assignment files already exist")

    p = sub.add_parser("map-proteins-to-families")
    add_common(p)
    p.add_argument("--proteins_fasta", required=True)
    p.add_argument("--subfamily_to_family_strict", required=True)
    p.add_argument("--subfamily_to_family_functional", required=True)
    p.add_argument("--subfamily_map", required=True)
    p.add_argument("--outdir", default="results/05_domain_hits")
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if protein mapping outputs already exist")

    p = sub.add_parser("write-matrices")
    add_common(p)
    p.add_argument("--subfamily_map", required=True)
    p.add_argument("--protein_family_segments", required=True)
    p.add_argument("--outdir", default="results/07_membership_matrices")
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if matrix output files already exist")

    p = sub.add_parser("qc-plots")
    add_common(p)
    p.add_argument("--resume", action="store_true",
                   help="Skip QC plot generation if the output directory already has plots")

    p = sub.add_parser("run-all")
    add_common(p)
    p.add_argument("--proteins_fasta", required=True)
    p.add_argument("--weights_path", required=True)
    p.add_argument("--resume", action="store_true",
                   help="Resume long-running stages where supported (hmm-hmm-edges)")
    p.add_argument("--hmm-mode", default=None, dest="hmm_mode",
                   choices=["pairwise", "db-search"],
                   help="HMM-HMM execution mode for run-all (overrides config)")
    p.add_argument("--shard-id", type=int, default=0, dest="shard_id",
                   help="Shard index for the HMM-HMM step in run-all")
    p.add_argument("--n-shards", type=int, default=1, dest="n_shards",
                   help="Total shards for the HMM-HMM step in run-all")

    args = ap.parse_args()
    cmd = ALIASES.get(args.cmd, args.cmd)
    cfg = load_config(args.config)
    logger = setup_logging(Path(args.results_root) / "logs", cmd)

    manifest_tools: dict[str, str] = {}
    try:
        manifest_tools.update(require_executables(["mmseqs"], cfg["tools"]))
        manifest_tools.update(require_executables(["hhmake"], cfg["tools"]))
        # Check hhalign or hhsearch+ffindex_build depending on HMM-HMM mode
        hmm_mode = cfg.get("hmm_hmm", {}).get("mode", "pairwise")
        if hmm_mode == "db-search":
            manifest_tools.update(require_executables(["hhsearch", "ffindex_build"], cfg["tools"]))
        else:
            manifest_tools.update(require_executables(["hhalign"], cfg["tools"]))
    except Exception as exc:
        logger.info("Runtime tool check deferred: %s", exc)
    else:
        logger.info("mmseqs version: %s", executable_version(manifest_tools["mmseqs"]))
        if "hhalign" in manifest_tools:
            logger.info("hhalign version: %s", executable_version(manifest_tools["hhalign"]))
        if "hhsearch" in manifest_tools:
            logger.info("hhsearch version: %s", executable_version(manifest_tools["hhsearch"]))
        logger.info("hhmake version: %s", executable_version(manifest_tools.get("hhmake", "hhmake")))

    # For single-step commands also write a log into the step's own output directory
    # so all outputs for a given step are co-located.
    if cmd != "run-all" and hasattr(args, "outdir"):
        add_step_log_handler(logger, args.outdir, cmd)
    elif cmd == "knn":
        # knn uses out_tsv instead of outdir
        add_step_log_handler(logger, Path(args.out_tsv).parent, cmd)

    if cmd == "mmseqs-cluster":
        manifest_tools.update(mmseqs_cluster(args.proteins_fasta, args.outdir, cfg, logger,
                                              resume=args.resume))
    elif cmd == "build-profiles":
        manifest_tools.update(build_profiles(args.proteins_fasta, args.subfamily_map, args.outdir, cfg, logger,
                                             resume=args.resume))
    elif cmd == "embed":
        embed(args.reps_fasta, args.outdir, cfg, args.weights_path, logger, resume=args.resume)
    elif cmd == "knn":
        knn(args.embeddings, args.ids, args.lengths, args.out_tsv, cfg, logger=logger, resume=args.resume)
    elif cmd == "hmm-hmm-edges":
        manifest_tools.update(hmm_hmm_edges(
            args.profile_index, args.outdir, cfg, logger, args.candidate_edges,
            mode=args.mode, resume=args.resume, shard_id=args.shard_id, n_shards=args.n_shards,
        ))
    elif cmd == "merge-hmm-shards":
        merge_hmm_shards(args.outdir, cfg, logger, resume=args.resume)
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
            logger=logger,
            resume=args.resume,
        )
    elif cmd == "cluster-families":
        cluster_families(args.merged_edges_strict, args.merged_edges_functional, args.subfamily_map, args.outdir, cfg,
                         args.method, logger=logger, resume=args.resume)
    elif cmd == "map-proteins-to-families":
        map_proteins_to_families(
            args.proteins_fasta,
            args.subfamily_to_family_strict,
            args.subfamily_to_family_functional,
            args.subfamily_map,
            args.outdir,
            cfg,
            logger,
            resume=args.resume,
        )
    elif cmd == "write-matrices":
        write_matrices(args.subfamily_map, args.protein_family_segments, args.outdir, cfg,
                       logger=logger, resume=args.resume)
    elif cmd == "qc-plots":
        generate_qc_plots(args.results_root, logger)
    elif cmd == "run-all":
        import time as _time
        root = Path(args.results_root)
        resume = args.resume
        logger.info("Starting run-all pipeline%s", " (resume mode)" if resume else "")

        def _timed_step(label, func, *a, step_log_dir=None, step_log_name=None, **kw):
            """Run a pipeline step, timing it and optionally logging to the step's output dir."""
            logger.info("Starting %s", label)
            t0 = _time.time()
            step_fh = None
            if step_log_dir and step_log_name:
                step_fh = add_step_log_handler(logger, step_log_dir, step_log_name)
            try:
                result = func(*a, **kw)
            finally:
                elapsed = _time.time() - t0
                logger.info("Completed %s in %.1fs", label, elapsed)
                if step_fh is not None:
                    logger.removeHandler(step_fh)
                    step_fh.close()
            if isinstance(result, dict):
                manifest_tools.update(result)
            return result

        _timed_step("Step 1/9: MMseqs clustering",
            mmseqs_cluster, args.proteins_fasta, str(root / "01_mmseqs"), cfg, logger,
            step_log_dir=root / "01_mmseqs", step_log_name="mmseqs-cluster",
            resume=resume)

        _timed_step("Step 2/9: Building profiles",
            build_profiles, args.proteins_fasta, str(root / "01_mmseqs/subfamily_map.tsv"),
            str(root / "02_profiles"), cfg, logger,
            step_log_dir=root / "02_profiles", step_log_name="build-profiles",
            resume=resume)

        _timed_step("Step 3/9: Embedding subfamily representatives",
            embed, str(root / "01_mmseqs/subfamily_reps.faa"), str(root / "04_embeddings"),
            cfg, args.weights_path, logger,
            step_log_dir=root / "04_embeddings", step_log_name="embed",
            resume=resume)

        _timed_step("Step 4/9: Computing KNN edges",
            knn, str(root / "04_embeddings/embeddings.npy"), str(root / "04_embeddings/ids.txt"),
            str(root / "04_embeddings/lengths.tsv"),
            str(root / "04_embeddings/embedding_knn_edges.tsv"),
            cfg,
            step_log_dir=root / "04_embeddings", step_log_name="knn",
            logger=logger, resume=resume)

        _timed_step("Step 5/9: Computing HMM-HMM edges",
            hmm_hmm_edges, str(root / "02_profiles/subfamily_profile_index.tsv"),
            str(root / "03_hmm_hmm_edges"), cfg, logger,
            str(root / "04_embeddings/embedding_knn_edges.tsv"),
            step_log_dir=root / "03_hmm_hmm_edges", step_log_name="hmm-hmm-edges",
            mode=getattr(args, "hmm_mode", None), resume=resume,
            shard_id=getattr(args, "shard_id", 0), n_shards=getattr(args, "n_shards", 1))

        _timed_step("Step 6/9: Merging graphs",
            merge_graph,
            str(root / "03_hmm_hmm_edges/hmm_hmm_edges_core.tsv"),
            str(root / "04_embeddings/embedding_knn_edges.tsv"),
            str(root / "06_family_clustering/merged_edges_strict.tsv"),
            str(root / "06_family_clustering/merged_edges_functional.tsv"),
            cfg,
            str(root / "03_hmm_hmm_edges/hmm_hmm_edges_relaxed.tsv"),
            step_log_dir=root / "06_family_clustering", step_log_name="merge-graph",
            logger=logger,
            resume=resume,
        )

        _timed_step("Step 7/9: Clustering families",
            cluster_families,
            str(root / "06_family_clustering/merged_edges_strict.tsv"),
            str(root / "06_family_clustering/merged_edges_functional.tsv"),
            str(root / "01_mmseqs/subfamily_map.tsv"),
            str(root / "06_family_clustering"),
            cfg,
            "leiden",
            step_log_dir=root / "06_family_clustering", step_log_name="cluster-families",
            logger=logger,
            resume=resume,
        )

        _timed_step("Step 8a/9: Mapping proteins to families",
            map_proteins_to_families,
            args.proteins_fasta,
            str(root / "06_family_clustering/subfamily_to_family_strict.tsv"),
            str(root / "06_family_clustering/subfamily_to_family_functional.tsv"),
            str(root / "01_mmseqs/subfamily_map.tsv"),
            str(root / "05_domain_hits"),
            cfg,
            logger,
            step_log_dir=root / "05_domain_hits", step_log_name="map-proteins-to-families",
            resume=resume,
        )

        _timed_step("Step 8b/9: Writing matrices",
            write_matrices,
            str(root / "01_mmseqs/subfamily_map.tsv"),
            str(root / "05_domain_hits/protein_family_segments.tsv"),
            str(root / "07_membership_matrices"),
            cfg,
            step_log_dir=root / "07_membership_matrices", step_log_name="write-matrices",
            logger=logger,
            resume=resume,
        )

        _timed_step("Step 9/9: Generating QC plots",
            generate_qc_plots, args.results_root, logger)

        logger.info("Pipeline completed successfully")

    write_manifest(
        Path(args.results_root) / "manifests" / "run_manifest.json",
        {"command": cmd, **vars(args)},
        manifest_tools,
        [getattr(args, "proteins_fasta", "")],
    )


if __name__ == "__main__":
    main()
