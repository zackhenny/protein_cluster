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
    orthofinder_cluster,
    write_matrices,
)
from .qc_plots import generate_qc_plots
from .runtime import add_step_log_handler, executable_version, require_executables, setup_logging, write_manifest


ALIASES = {
    "map-proteins": "map-proteins-to-families",
    "cluster": "cluster-families",
}


def add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--config", default=None,
                   help="Path to a YAML config file.  All paths (results_root, "
                        "proteins_fasta, weights_path) can be set there so that "
                        "individual module commands work with --config alone.")
    # Default None so we can detect whether the user explicitly provided this
    # and fall back to the config value if not.
    p.add_argument("--results_root", default=None,
                   help="Top-level output directory (overrides config results_root). "
                        "Default: value of results_root in config, or 'results'.")


def _require(value: "str | None", name: str) -> str:
    """Raise a clear error when a required path was not resolved."""
    if value:
        return value
    raise SystemExit(
        f"ERROR: required argument '{name}' was not provided on the command line "
        f"and could not be derived from the config.\n"
        f"Either pass it explicitly or set 'results_root' (and, if needed, "
        f"'proteins_fasta' / 'weights_path') in your config file."
    )

def main() -> None:
    ap = argparse.ArgumentParser(prog="plm_cluster")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("mmseqs-cluster")
    add_common(p)
    p.add_argument("--proteins_fasta", default=None,
                   help="Input protein FASTA. Defaults to config proteins_fasta.")
    p.add_argument("--outdir", default=None,
                   help="Output directory. Defaults to {results_root}/01_mmseqs.")
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if its output files already exist")

    p = sub.add_parser("build-profiles")
    add_common(p)
    p.add_argument("--proteins_fasta", default=None,
                   help="Input protein FASTA. Defaults to config proteins_fasta.")
    p.add_argument("--subfamily_map", default=None,
                   help="Subfamily map TSV. Defaults to {results_root}/01_mmseqs/subfamily_map.tsv.")
    p.add_argument("--outdir", default=None,
                   help="Output directory. Defaults to {results_root}/02_profiles.")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-built per-subfamily profiles; rebuild only missing ones")

    p = sub.add_parser("embed")
    add_common(p)
    p.add_argument("--reps_fasta", default=None,
                   help="Subfamily representative FASTA. Defaults to {results_root}/01_mmseqs/subfamily_reps.faa.")
    p.add_argument("--weights_path", default=None,
                   help="ESM-2 weights directory. Defaults to config weights_path.")
    p.add_argument("--outdir", default=None,
                   help="Output directory. Defaults to {results_root}/04_embeddings.")
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if embeddings.npy already exists")

    p = sub.add_parser("knn")
    add_common(p)
    p.add_argument("--embeddings", default=None,
                   help="Embeddings .npy file. Defaults to {results_root}/04_embeddings/embeddings.npy.")
    p.add_argument("--ids", default=None,
                   help="IDs text file. Defaults to {results_root}/04_embeddings/ids.txt.")
    p.add_argument("--lengths", default=None,
                   help="Lengths TSV. Defaults to {results_root}/04_embeddings/lengths.tsv.")
    p.add_argument("--out_tsv", default=None,
                   help="Output edge TSV. Defaults to {results_root}/04_embeddings/embedding_knn_edges.tsv.")
    p.add_argument("--subfamily_map", default=None,
                   help="Subfamily map TSV (optional; used for rkcnn class labels). "
                        "Defaults to {results_root}/01_mmseqs/subfamily_map.tsv.")
    p.add_argument("--mode", default=None, choices=["knn", "rkcnn"],
                   help="Candidate generation mode: 'knn' (default) or 'rkcnn'")
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if the output TSV already exists")

    p = sub.add_parser("hmm-hmm-edges")
    add_common(p)
    p.add_argument("--profile_index", default=None,
                   help="Profile index TSV. Defaults to {results_root}/02_profiles/subfamily_profile_index.tsv.")
    p.add_argument("--candidate_edges", default=None,
                   help="KNN edge TSV. Defaults to {results_root}/04_embeddings/embedding_knn_edges.tsv.")
    p.add_argument("--mode", default=None, choices=["pairwise", "db-search", "mmseqs-profile"],
                   help="Execution mode: 'pairwise' (default), 'db-search' (hhsearch DB search), "
                        "or 'mmseqs-profile' (MMseqs2 profile-profile search, fastest for large datasets)")
    p.add_argument("--resume", action="store_true",
                   help="Skip already-completed pairs using the NDJSON progress log")
    p.add_argument("--shard-id", type=int, default=0, dest="shard_id",
                   help="Zero-based shard index for parallel execution (default: 0)")
    p.add_argument("--n-shards", type=int, default=1, dest="n_shards",
                   help="Total number of shards (default: 1 = no sharding)")
    p.add_argument("--outdir", default=None,
                   help="Output directory. Defaults to {results_root}/03_hmm_hmm_edges.")

    p = sub.add_parser("merge-hmm-shards")
    add_common(p)
    p.add_argument("--outdir", default=None,
                   help="Directory containing per-shard TSV files to merge. "
                        "Defaults to {results_root}/03_hmm_hmm_edges.")
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if the merged output already exists")

    p = sub.add_parser("merge-graph")
    add_common(p)
    p.add_argument("--hmm_core", default=None,
                   help="HMM core edges TSV. Defaults to {results_root}/03_hmm_hmm_edges/hmm_hmm_edges_core.tsv.")
    p.add_argument("--embedding_edges", default=None,
                   help="KNN embedding edges TSV. Defaults to {results_root}/04_embeddings/embedding_knn_edges.tsv.")
    p.add_argument("--hmm_relaxed", default=None,
                   help="HMM relaxed edges TSV. Defaults to {results_root}/03_hmm_hmm_edges/hmm_hmm_edges_relaxed.tsv.")
    p.add_argument("--outdir", default=None,
                   help="Output directory. Defaults to {results_root}/06_family_clustering.")
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if merged graph files already exist")

    p = sub.add_parser("cluster-families")
    add_common(p)
    p.add_argument("--merged_edges_strict", default=None,
                   help="Strict merged edges TSV. Defaults to {results_root}/06_family_clustering/merged_edges_strict.tsv.")
    p.add_argument("--merged_edges_functional", default=None,
                   help="Functional merged edges TSV. Defaults to {results_root}/06_family_clustering/merged_edges_functional.tsv.")
    p.add_argument("--subfamily_map", default=None,
                   help="Subfamily map TSV. Defaults to {results_root}/01_mmseqs/subfamily_map.tsv.")
    p.add_argument("--outdir", default=None,
                   help="Output directory. Defaults to {results_root}/06_family_clustering.")
    p.add_argument("--method", default="leiden", choices=["leiden", "mcl"])
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if family assignment files already exist")

    p = sub.add_parser("map-proteins-to-families")
    add_common(p)
    p.add_argument("--proteins_fasta", default=None,
                   help="Input protein FASTA. Defaults to config proteins_fasta.")
    p.add_argument("--subfamily_to_family_strict", default=None,
                   help="Strict mapping TSV. Defaults to {results_root}/06_family_clustering/subfamily_to_family_strict.tsv.")
    p.add_argument("--subfamily_to_family_functional", default=None,
                   help="Functional mapping TSV. Defaults to {results_root}/06_family_clustering/subfamily_to_family_functional.tsv.")
    p.add_argument("--subfamily_map", default=None,
                   help="Subfamily map TSV. Defaults to {results_root}/01_mmseqs/subfamily_map.tsv.")
    p.add_argument("--outdir", default=None,
                   help="Output directory. Defaults to {results_root}/05_domain_hits.")
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if protein mapping outputs already exist")

    p = sub.add_parser("write-matrices")
    add_common(p)
    p.add_argument("--subfamily_map", default=None,
                   help="Subfamily map TSV. Defaults to {results_root}/01_mmseqs/subfamily_map.tsv.")
    p.add_argument("--protein_family_segments", default=None,
                   help="Protein-family segments TSV. Defaults to {results_root}/05_domain_hits/protein_family_segments.tsv.")
    p.add_argument("--outdir", default=None,
                   help="Output directory. Defaults to {results_root}/07_membership_matrices.")
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if matrix output files already exist")

    p = sub.add_parser("qc-plots")
    add_common(p)
    p.add_argument("--resume", action="store_true",
                   help="Skip QC plot generation if the output directory already has plots")

    p = sub.add_parser("run-all")
    add_common(p)
    p.add_argument("--proteins_fasta", default=None,
                   help="Input protein FASTA. Defaults to config proteins_fasta.")
    p.add_argument("--weights_path", default=None,
                   help="ESM-2 weights directory. Defaults to config weights_path.")
    p.add_argument("--resume", action="store_true",
                   help="Resume long-running stages where supported (hmm-hmm-edges)")
    p.add_argument("--hmm-mode", default=None, dest="hmm_mode",
                   choices=["pairwise", "db-search", "mmseqs-profile"],
                   help="HMM-HMM execution mode for run-all (overrides config)")
    p.add_argument("--knn-mode", default=None, dest="knn_mode",
                   choices=["knn", "rkcnn"],
                   help="KNN candidate generation mode for run-all (overrides config)")
    p.add_argument("--shard-id", type=int, default=0, dest="shard_id",
                   help="Shard index for the HMM-HMM step in run-all")
    p.add_argument("--n-shards", type=int, default=1, dest="n_shards",
                   help="Total shards for the HMM-HMM step in run-all")

    p = sub.add_parser("orthofinder-cluster")
    add_common(p)
    p.add_argument("--og_dir", required=True,
                   help="Directory of OrthoFinder HOG or OG *.faa / *.fa files to subcluster")
    p.add_argument("--outdir", default=None,
                   help="Output directory. Defaults to {results_root}/01_mmseqs.")
    p.add_argument("--resume", action="store_true",
                   help="Skip this step if its output files already exist")
    p.add_argument("--gene-trees-source", default=None, dest="gene_trees_source",
                   help="Path to a Resolved_Gene_Trees.txt file or a directory of "
                        "*_tree.txt files (OrthoFinder v3); only OGs with a gene tree "
                        "are processed")

    p = sub.add_parser("run-all-orthofinder")
    add_common(p)
    p.add_argument("--og_dir", required=True,
                   help="Directory of OrthoFinder HOG or OG *.faa / *.fa files "
                        "(e.g. OrthoFinder/Results_*/Phylogenetic_Hierarchical_Orthogroups/N0/)")
    p.add_argument("--weights_path", default=None,
                   help="ESM-2 weights directory. Defaults to config weights_path.")
    p.add_argument("--resume", action="store_true",
                   help="Resume long-running stages where supported (hmm-hmm-edges)")
    p.add_argument("--hmm-mode", default=None, dest="hmm_mode",
                   choices=["pairwise", "db-search", "mmseqs-profile"],
                   help="HMM-HMM execution mode (overrides config)")
    p.add_argument("--knn-mode", default=None, dest="knn_mode",
                   choices=["knn", "rkcnn"],
                   help="KNN candidate generation mode (overrides config)")
    p.add_argument("--shard-id", type=int, default=0, dest="shard_id",
                   help="Shard index for the HMM-HMM step")
    p.add_argument("--n-shards", type=int, default=1, dest="n_shards",
                   help="Total shards for the HMM-HMM step")
    p.add_argument("--gene-trees-source", default=None, dest="gene_trees_source",
                   help="Path to a Resolved_Gene_Trees.txt file or a directory of "
                        "*_tree.txt files (OrthoFinder v3); only OGs with a gene tree "
                        "are processed")

    args = ap.parse_args()
    cmd = ALIASES.get(args.cmd, args.cmd)
    cfg = load_config(args.config)

    # Resolve results_root: CLI flag > config value > hardcoded default.
    results_root_str: str = args.results_root or cfg.get("results_root", "results")
    root = Path(results_root_str)

    # Build a helper that resolves all step input/output paths from results_root.
    # Each value is: CLI arg (if provided) falling back to the derived path.
    def _p(*parts: str) -> str:
        """Return a path string rooted at results_root."""
        return str(root.joinpath(*parts))

    logger = setup_logging(root / "logs", cmd)

    def _resolve_gene_trees_source() -> "str | None":
        """Return the effective gene_trees_source: CLI flag > config > None."""
        cli_val = getattr(args, "gene_trees_source", None)
        return cli_val or cfg.get("orthofinder", {}).get("gene_trees_source") or None

    # Determine effective HMM mode BEFORE tool preflight checks so that
    # mmseqs-profile mode is not incorrectly required to have hhalign, and
    # db-search mode correctly requires hhsearch+ffindex_build+cstranslate.
    _cli_hmm_mode = getattr(args, "hmm_mode", None)
    effective_hmm_mode = _cli_hmm_mode or cfg.get("hmm_hmm", {}).get("mode", "pairwise")
    # Propagate CLI override into the config so downstream pipeline steps see it.
    if _cli_hmm_mode:
        cfg["hmm_hmm"]["mode"] = _cli_hmm_mode

    # Propagate --knn-mode CLI override into the config so the KNN step sees it.
    _cli_knn_mode = getattr(args, "knn_mode", None)
    if _cli_knn_mode:
        cfg["knn"]["mode"] = _cli_knn_mode

    manifest_tools: dict[str, str] = {}
    try:
        manifest_tools.update(require_executables(["mmseqs"], cfg["tools"]))
        manifest_tools.update(require_executables(["hhmake"], cfg["tools"]))
        # Check tools required by the *effective* HMM mode.
        # mmseqs-profile only needs mmseqs (already checked above).
        if effective_hmm_mode == "db-search":
            manifest_tools.update(require_executables(["hhsearch", "ffindex_build", "cstranslate"], cfg["tools"]))
        elif effective_hmm_mode == "pairwise":
            manifest_tools.update(require_executables(["hhalign"], cfg["tools"]))
        # mmseqs-profile: no additional tool requirement beyond mmseqs
    except Exception as exc:
        logger.info("Runtime tool check deferred: %s", exc)
    else:
        logger.info("mmseqs version: %s", executable_version(manifest_tools["mmseqs"]))
        if "hhalign" in manifest_tools:
            logger.info("hhalign version: %s", executable_version(manifest_tools["hhalign"]))
        if "hhsearch" in manifest_tools:
            logger.info("hhsearch version: %s", executable_version(manifest_tools["hhsearch"]))
        logger.info("hhmake version: %s", executable_version(manifest_tools.get("hhmake", "hhmake")))

    # -----------------------------------------------------------------------
    # Resolve per-step paths: CLI value > derived from results_root.
    # -----------------------------------------------------------------------
    # Input files that must be explicitly provided or set in config
    cfg_proteins_fasta: str = cfg.get("proteins_fasta", "")
    cfg_weights_path: str = cfg.get("weights_path", "")

    # Helper: return arg value if set, else fallback, else error
    def _arg_or(arg_val: "str | None", fallback: str, name: str) -> str:
        if arg_val:
            return arg_val
        if fallback:
            return fallback
        return _require(None, name)

    # For single-step commands also write a log into the step's own output directory
    # so all outputs for a given step are co-located.
    # We resolve outdir before setting up the step log handler.
    _step_outdir: "str | None" = getattr(args, "outdir", None)

    if cmd == "mmseqs-cluster":
        outdir = _step_outdir or _p("01_mmseqs")
        add_step_log_handler(logger, outdir, cmd)
        manifest_tools.update(mmseqs_cluster(
            _arg_or(args.proteins_fasta, cfg_proteins_fasta, "--proteins_fasta"),
            outdir, cfg, logger, resume=args.resume))

    elif cmd == "build-profiles":
        outdir = _step_outdir or _p("02_profiles")
        add_step_log_handler(logger, outdir, cmd)
        manifest_tools.update(build_profiles(
            _arg_or(args.proteins_fasta, cfg_proteins_fasta, "--proteins_fasta"),
            args.subfamily_map or _p("01_mmseqs", "subfamily_map.tsv"),
            outdir, cfg, logger, resume=args.resume))

    elif cmd == "embed":
        outdir = _step_outdir or _p("04_embeddings")
        add_step_log_handler(logger, outdir, cmd)
        embed(
            args.reps_fasta or _p("01_mmseqs", "subfamily_reps.faa"),
            outdir, cfg,
            _arg_or(args.weights_path, cfg_weights_path, "--weights_path"),
            logger, resume=args.resume)

    elif cmd == "knn":
        out_tsv = args.out_tsv or _p("04_embeddings", "embedding_knn_edges.tsv")
        add_step_log_handler(logger, Path(out_tsv).parent, cmd)
        # Override knn.mode from CLI if provided
        if getattr(args, "mode", None):
            cfg["knn"]["mode"] = args.mode
        knn(
            args.embeddings or _p("04_embeddings", "embeddings.npy"),
            args.ids or _p("04_embeddings", "ids.txt"),
            args.lengths or _p("04_embeddings", "lengths.tsv"),
            out_tsv, cfg, logger=logger, resume=args.resume,
            subfamily_map=args.subfamily_map or _p("01_mmseqs", "subfamily_map.tsv"))

    elif cmd == "hmm-hmm-edges":
        outdir = _step_outdir or _p("03_hmm_hmm_edges")
        add_step_log_handler(logger, outdir, cmd)
        manifest_tools.update(hmm_hmm_edges(
            args.profile_index or _p("02_profiles", "subfamily_profile_index.tsv"),
            outdir, cfg, logger,
            args.candidate_edges or _p("04_embeddings", "embedding_knn_edges.tsv"),
            mode=args.mode, resume=args.resume,
            shard_id=args.shard_id, n_shards=args.n_shards))

    elif cmd == "merge-hmm-shards":
        outdir = _step_outdir or _p("03_hmm_hmm_edges")
        add_step_log_handler(logger, outdir, cmd)
        merge_hmm_shards(outdir, cfg, logger, resume=args.resume)

    elif cmd == "merge-graph":
        outdir = _step_outdir or _p("06_family_clustering")
        add_step_log_handler(logger, outdir, cmd)
        out = Path(outdir)
        out.mkdir(parents=True, exist_ok=True)
        merge_graph(
            args.hmm_core or _p("03_hmm_hmm_edges", "hmm_hmm_edges_core.tsv"),
            args.embedding_edges or _p("04_embeddings", "embedding_knn_edges.tsv"),
            str(out / "merged_edges_strict.tsv"),
            str(out / "merged_edges_functional.tsv"),
            cfg,
            args.hmm_relaxed or _p("03_hmm_hmm_edges", "hmm_hmm_edges_relaxed.tsv"),
            logger=logger, resume=args.resume)

    elif cmd == "cluster-families":
        outdir = _step_outdir or _p("06_family_clustering")
        add_step_log_handler(logger, outdir, cmd)
        cluster_families(
            args.merged_edges_strict or _p("06_family_clustering", "merged_edges_strict.tsv"),
            args.merged_edges_functional or _p("06_family_clustering", "merged_edges_functional.tsv"),
            args.subfamily_map or _p("01_mmseqs", "subfamily_map.tsv"),
            outdir, cfg, args.method, logger=logger, resume=args.resume)

    elif cmd == "map-proteins-to-families":
        outdir = _step_outdir or _p("05_domain_hits")
        add_step_log_handler(logger, outdir, cmd)
        map_proteins_to_families(
            _arg_or(args.proteins_fasta, cfg_proteins_fasta, "--proteins_fasta"),
            args.subfamily_to_family_strict or _p("06_family_clustering", "subfamily_to_family_strict.tsv"),
            args.subfamily_to_family_functional or _p("06_family_clustering", "subfamily_to_family_functional.tsv"),
            args.subfamily_map or _p("01_mmseqs", "subfamily_map.tsv"),
            outdir, cfg, logger, resume=args.resume)

    elif cmd == "write-matrices":
        outdir = _step_outdir or _p("07_membership_matrices")
        add_step_log_handler(logger, outdir, cmd)
        write_matrices(
            args.subfamily_map or _p("01_mmseqs", "subfamily_map.tsv"),
            args.protein_family_segments or _p("05_domain_hits", "protein_family_segments.tsv"),
            outdir, cfg, logger=logger, resume=args.resume)

    elif cmd == "qc-plots":
        add_step_log_handler(logger, root / "qc_plots", cmd)
        generate_qc_plots(results_root_str, logger, resume=args.resume)

    elif cmd == "run-all":
        import time as _time
        resume = args.resume
        proteins_fasta = _arg_or(args.proteins_fasta, cfg_proteins_fasta, "--proteins_fasta")
        weights_path = _arg_or(args.weights_path, cfg_weights_path, "--weights_path")
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
            mmseqs_cluster, proteins_fasta, _p("01_mmseqs"), cfg, logger,
            step_log_dir=root / "01_mmseqs", step_log_name="mmseqs-cluster",
            resume=resume)

        _timed_step("Step 2/9: Building profiles",
            build_profiles, proteins_fasta, _p("01_mmseqs", "subfamily_map.tsv"),
            _p("02_profiles"), cfg, logger,
            step_log_dir=root / "02_profiles", step_log_name="build-profiles",
            resume=resume)

        _timed_step("Step 3/9: Embedding subfamily representatives",
            embed, _p("01_mmseqs", "subfamily_reps.faa"), _p("04_embeddings"),
            cfg, weights_path, logger,
            step_log_dir=root / "04_embeddings", step_log_name="embed",
            resume=resume)

        _timed_step("Step 4/9: Computing KNN edges",
            knn, _p("04_embeddings", "embeddings.npy"), _p("04_embeddings", "ids.txt"),
            _p("04_embeddings", "lengths.tsv"),
            _p("04_embeddings", "embedding_knn_edges.tsv"),
            cfg,
            step_log_dir=root / "04_embeddings", step_log_name="knn",
            logger=logger, resume=resume,
            subfamily_map=_p("01_mmseqs", "subfamily_map.tsv"))

        _timed_step("Step 5/9: Computing HMM-HMM edges",
            hmm_hmm_edges, _p("02_profiles", "subfamily_profile_index.tsv"),
            _p("03_hmm_hmm_edges"), cfg, logger,
            _p("04_embeddings", "embedding_knn_edges.tsv"),
            step_log_dir=root / "03_hmm_hmm_edges", step_log_name="hmm-hmm-edges",
            mode=getattr(args, "hmm_mode", None), resume=resume,
            shard_id=getattr(args, "shard_id", 0), n_shards=getattr(args, "n_shards", 1))

        _timed_step("Step 6/9: Merging graphs",
            merge_graph,
            _p("03_hmm_hmm_edges", "hmm_hmm_edges_core.tsv"),
            _p("04_embeddings", "embedding_knn_edges.tsv"),
            _p("06_family_clustering", "merged_edges_strict.tsv"),
            _p("06_family_clustering", "merged_edges_functional.tsv"),
            cfg,
            _p("03_hmm_hmm_edges", "hmm_hmm_edges_relaxed.tsv"),
            step_log_dir=root / "06_family_clustering", step_log_name="merge-graph",
            logger=logger,
            resume=resume,
        )

        _timed_step("Step 7/9: Clustering families",
            cluster_families,
            _p("06_family_clustering", "merged_edges_strict.tsv"),
            _p("06_family_clustering", "merged_edges_functional.tsv"),
            _p("01_mmseqs", "subfamily_map.tsv"),
            _p("06_family_clustering"),
            cfg,
            "leiden",
            step_log_dir=root / "06_family_clustering", step_log_name="cluster-families",
            logger=logger,
            resume=resume,
        )

        _timed_step("Step 8a/9: Mapping proteins to families",
            map_proteins_to_families,
            proteins_fasta,
            _p("06_family_clustering", "subfamily_to_family_strict.tsv"),
            _p("06_family_clustering", "subfamily_to_family_functional.tsv"),
            _p("01_mmseqs", "subfamily_map.tsv"),
            _p("05_domain_hits"),
            cfg,
            logger,
            step_log_dir=root / "05_domain_hits", step_log_name="map-proteins-to-families",
            resume=resume,
        )

        _timed_step("Step 8b/9: Writing matrices",
            write_matrices,
            _p("01_mmseqs", "subfamily_map.tsv"),
            _p("05_domain_hits", "protein_family_segments.tsv"),
            _p("07_membership_matrices"),
            cfg,
            step_log_dir=root / "07_membership_matrices", step_log_name="write-matrices",
            logger=logger,
            resume=resume,
        )

        _timed_step("Step 9/9: Generating QC plots",
            generate_qc_plots, results_root_str, logger,
            step_log_dir=root / "qc_plots", step_log_name="qc-plots",
            resume=resume)

        logger.info("Pipeline completed successfully")

    elif cmd == "orthofinder-cluster":
        outdir = _step_outdir or _p("01_mmseqs")
        add_step_log_handler(logger, outdir, cmd)
        manifest_tools.update(orthofinder_cluster(
            args.og_dir, outdir, cfg, logger,
            resume=args.resume,
            gene_trees_source=_resolve_gene_trees_source()))

    elif cmd == "run-all-orthofinder":
        import time as _time
        resume = args.resume
        weights_path = _arg_or(args.weights_path, cfg_weights_path, "--weights_path")
        logger.info("Starting run-all-orthofinder pipeline%s", " (resume mode)" if resume else "")

        def _timed_step_of(label, func, *a, step_log_dir=None, step_log_name=None, **kw):
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

        _timed_step_of("Step 1/9: OrthoFinder subclustering",
            orthofinder_cluster, args.og_dir, _p("01_mmseqs"), cfg, logger,
            step_log_dir=root / "01_mmseqs", step_log_name="orthofinder-cluster",
            resume=resume,
            gene_trees_source=_resolve_gene_trees_source())

        proteins_combined = _p("01_mmseqs", "proteins_combined.faa")

        _timed_step_of("Step 2/9: Building profiles",
            build_profiles, proteins_combined, _p("01_mmseqs", "subfamily_map.tsv"),
            _p("02_profiles"), cfg, logger,
            step_log_dir=root / "02_profiles", step_log_name="build-profiles",
            resume=resume)

        _timed_step_of("Step 3/9: Embedding subfamily representatives",
            embed, _p("01_mmseqs", "subfamily_reps.faa"), _p("04_embeddings"),
            cfg, weights_path, logger,
            step_log_dir=root / "04_embeddings", step_log_name="embed",
            resume=resume)

        _timed_step_of("Step 4/9: Computing KNN edges",
            knn, _p("04_embeddings", "embeddings.npy"), _p("04_embeddings", "ids.txt"),
            _p("04_embeddings", "lengths.tsv"),
            _p("04_embeddings", "embedding_knn_edges.tsv"),
            cfg,
            step_log_dir=root / "04_embeddings", step_log_name="knn",
            logger=logger, resume=resume,
            subfamily_map=_p("01_mmseqs", "subfamily_map.tsv"))

        _timed_step_of("Step 5/9: Computing HMM-HMM edges",
            hmm_hmm_edges, _p("02_profiles", "subfamily_profile_index.tsv"),
            _p("03_hmm_hmm_edges"), cfg, logger,
            _p("04_embeddings", "embedding_knn_edges.tsv"),
            step_log_dir=root / "03_hmm_hmm_edges", step_log_name="hmm-hmm-edges",
            mode=getattr(args, "hmm_mode", None), resume=resume,
            shard_id=getattr(args, "shard_id", 0), n_shards=getattr(args, "n_shards", 1))

        _timed_step_of("Step 6/9: Merging graphs",
            merge_graph,
            _p("03_hmm_hmm_edges", "hmm_hmm_edges_core.tsv"),
            _p("04_embeddings", "embedding_knn_edges.tsv"),
            _p("06_family_clustering", "merged_edges_strict.tsv"),
            _p("06_family_clustering", "merged_edges_functional.tsv"),
            cfg,
            _p("03_hmm_hmm_edges", "hmm_hmm_edges_relaxed.tsv"),
            step_log_dir=root / "06_family_clustering", step_log_name="merge-graph",
            logger=logger,
            resume=resume,
        )

        _timed_step_of("Step 7/9: Clustering families",
            cluster_families,
            _p("06_family_clustering", "merged_edges_strict.tsv"),
            _p("06_family_clustering", "merged_edges_functional.tsv"),
            _p("01_mmseqs", "subfamily_map.tsv"),
            _p("06_family_clustering"),
            cfg,
            "leiden",
            step_log_dir=root / "06_family_clustering", step_log_name="cluster-families",
            logger=logger,
            resume=resume,
        )

        _timed_step_of("Step 8a/9: Mapping proteins to families",
            map_proteins_to_families,
            proteins_combined,
            _p("06_family_clustering", "subfamily_to_family_strict.tsv"),
            _p("06_family_clustering", "subfamily_to_family_functional.tsv"),
            _p("01_mmseqs", "subfamily_map.tsv"),
            _p("05_domain_hits"),
            cfg,
            logger,
            step_log_dir=root / "05_domain_hits", step_log_name="map-proteins-to-families",
            resume=resume,
        )

        _timed_step_of("Step 8b/9: Writing matrices",
            write_matrices,
            _p("01_mmseqs", "subfamily_map.tsv"),
            _p("05_domain_hits", "protein_family_segments.tsv"),
            _p("07_membership_matrices"),
            cfg,
            step_log_dir=root / "07_membership_matrices", step_log_name="write-matrices",
            logger=logger,
            resume=resume,
        )

        _timed_step_of("Step 9/9: Generating QC plots",
            generate_qc_plots, results_root_str, logger,
            step_log_dir=root / "qc_plots", step_log_name="qc-plots",
            resume=resume)

        logger.info("Pipeline completed successfully")

    write_manifest(
        root / "manifests" / "run_manifest.json",
        {"command": cmd, **vars(args), "results_root": results_root_str},
        manifest_tools,
        [getattr(args, "og_dir", None) or getattr(args, "proteins_fasta", "") or cfg_proteins_fasta],
    )


if __name__ == "__main__":
    main()
