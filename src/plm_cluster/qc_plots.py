"""QC plots for plm-cluster pipeline outputs.

Generates diagnostic figures from pipeline intermediate files and saves them
to ``results/qc_plots/``.  Requires matplotlib; gracefully skips if missing.

MMseqs2 clustering step plots are generated separately via
``generate_mmseqs_cluster_plots`` using the plotnine library and saved
directly into the mmseqs-cluster output folder (``results/01_mmseqs/plots/``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


def _safe_read(path: str | Path, **kw) -> pd.DataFrame | None:
    """Read a TSV if it exists and is non-empty, else return None."""
    p = Path(path)
    if p.exists() and p.stat().st_size > 0:
        return pd.read_csv(p, sep="\t", **kw)
    return None


# ---------------------------------------------------------------------------
# Individual matplotlib plot functions (full-pipeline QC dashboard)
# ---------------------------------------------------------------------------

def plot_subfamily_sizes(results_root: str, ax: Any) -> None:
    """Histogram of subfamily member counts (from step 01)."""
    df = _safe_read(Path(results_root) / "01_mmseqs" / "subfamily_stats.tsv")
    if df is None:
        ax.set_visible(False)
        return
    sizes = df["n_members"]
    ax.hist(sizes, bins=min(50, max(10, len(sizes) // 5)), color="#4c72b0", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Members per subfamily")
    ax.set_ylabel("Count")
    ax.set_title("Subfamily size distribution")
    ax.set_yscale("log")
    median = float(np.median(sizes))
    ax.axvline(median, color="#c44e52", linestyle="--", label=f"median = {median:.0f}")
    ax.legend(fontsize=8)


def plot_hmm_edge_weights(results_root: str, ax: Any) -> None:
    """Overlapping histograms of prob and edge_weight from HMM-HMM edges."""
    df = _safe_read(Path(results_root) / "03_hmm_hmm_edges" / "hmm_hmm_edges_raw.tsv")
    if df is None:
        ax.set_visible(False)
        return
    if "prob" in df.columns:
        ax.hist(df["prob"], bins=50, alpha=0.6, label="Probability", color="#4c72b0", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("HHalign probability")
    ax.set_ylabel("Count")
    ax.set_title("HMM-HMM edge probability distribution")
    ax.legend(fontsize=8)


def plot_embedding_cosines(results_root: str, ax: Any) -> None:
    """Histogram of cosine similarities from KNN embedding edges."""
    df = _safe_read(Path(results_root) / "04_embeddings" / "embedding_knn_edges.tsv")
    if df is None:
        ax.set_visible(False)
        return
    ax.hist(df["cosine"], bins=50, color="#55a868", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Cosine similarity")
    ax.set_ylabel("Count")
    ax.set_title("Embedding KNN cosine distribution")
    median = float(np.median(df["cosine"]))
    ax.axvline(median, color="#c44e52", linestyle="--", label=f"median = {median:.2f}")
    ax.legend(fontsize=8)


def plot_family_sizes(results_root: str, ax: Any) -> None:
    """Side-by-side subfamily counts for strict vs functional families."""
    s_df = _safe_read(Path(results_root) / "06_family_clustering" / "family_stats_strict.tsv")
    f_df = _safe_read(Path(results_root) / "06_family_clustering" / "family_stats_functional.tsv")
    if s_df is None and f_df is None:
        ax.set_visible(False)
        return
    if s_df is not None and "n_subfamilies" in s_df.columns:
        ax.hist(s_df["n_subfamilies"], bins=30, alpha=0.6, label="Strict", color="#4c72b0", edgecolor="white", linewidth=0.3)
    if f_df is not None and "n_subfamilies" in f_df.columns:
        ax.hist(f_df["n_subfamilies"], bins=30, alpha=0.6, label="Functional", color="#dd8452", edgecolor="white", linewidth=0.3)
    ax.set_xlabel("Subfamilies per family")
    ax.set_ylabel("Count")
    ax.set_title("Family size distribution")
    ax.set_yscale("log")
    ax.legend(fontsize=8)


def plot_protein_coverage(results_root: str, ax: Any) -> None:
    """Histogram of per-protein coverage fractions from domain mapping."""
    df = _safe_read(Path(results_root) / "05_domain_hits" / "protein_architectures.tsv")
    if df is None:
        ax.set_visible(False)
        return
    cov = df["coverage_fraction"]
    ax.hist(cov, bins=50, color="#8172b3", edgecolor="white", linewidth=0.5)
    ax.set_xlabel("Coverage fraction")
    ax.set_ylabel("Count")
    ax.set_title("Protein coverage by mapped families")
    mean_cov = float(np.mean(cov))
    ax.axvline(mean_cov, color="#c44e52", linestyle="--", label=f"mean = {mean_cov:.2f}")
    ax.legend(fontsize=8)


def plot_fusion_fraction(results_root: str, ax: Any) -> None:
    """Bar chart showing fraction of single-domain vs multi-domain (fusion) proteins."""
    df = _safe_read(Path(results_root) / "05_domain_hits" / "protein_architectures.tsv")
    if df is None:
        ax.set_visible(False)
        return
    n_total = len(df)
    n_mapped = int((df["n_segments"] > 0).sum())
    n_fusion = int((df["is_fusion"] == 1).sum())
    n_single = n_mapped - n_fusion
    n_unmapped = n_total - n_mapped

    labels = ["Unmapped", "Single-domain", "Multi-domain"]
    counts = [n_unmapped, n_single, n_fusion]
    colors = ["#cccccc", "#55a868", "#c44e52"]
    ax.bar(labels, counts, color=colors, edgecolor="white")
    ax.set_ylabel("Protein count")
    ax.set_title("Domain architecture summary")
    for i, v in enumerate(counts):
        ax.text(i, v + 0.01 * max(counts), str(v), ha="center", fontsize=8)


def plot_singleton_summary(results_root: str, ax: Any) -> None:
    """Bar chart of cluster size distribution with singleton percentage statistics in title."""
    report = _safe_read(Path(results_root) / "01_mmseqs" / "mmseqs_cluster_report.tsv")
    df = _safe_read(Path(results_root) / "01_mmseqs" / "subfamily_stats.tsv")

    if report is None:
        # Compute everything from subfamily_stats.tsv
        if df is None or "n_members" not in df.columns:
            ax.set_visible(False)
            return
        n_singletons = int((df["n_members"] == 1).sum())
        n_2 = int((df["n_members"] == 2).sum())
        n_3plus = int((df["n_members"] >= 3).sum())
    else:
        n_singletons = int(report["n_singletons"].iloc[0])
        n_total = int(report["n_clusters_total"].iloc[0])
        if df is not None and "n_members" in df.columns:
            n_2 = int((df["n_members"] == 2).sum())
            n_3plus = int((df["n_members"] >= 3).sum())
        else:
            n_2plus = n_total - n_singletons
            n_2 = 0
            n_3plus = n_2plus

    n_clusters_total = n_singletons + n_2 + n_3plus

    # Total proteins = sum of all members across all clusters
    if df is not None and "n_members" in df.columns:
        n_proteins_total = int(df["n_members"].sum())
        n_proteins_in_singletons = n_singletons  # each singleton cluster has exactly 1 protein
    else:
        n_proteins_total = None
        n_proteins_in_singletons = n_singletons

    categories = ["Singletons\n(n=1)", "2-member\nclusters", "\u22653-member\nclusters"]
    counts = [n_singletons, n_2, n_3plus]
    colors = ["#dd8452", "#4c72b0", "#55a868"]
    bars = ax.bar(categories, counts, color=colors, edgecolor="white")
    ax.set_ylabel("Cluster count")

    # Build title with singleton percentages
    if n_clusters_total > 0:
        pct_clusters = 100.0 * n_singletons / n_clusters_total
        if n_proteins_total and n_proteins_total > 0:
            pct_proteins = 100.0 * n_proteins_in_singletons / n_proteins_total
            ax.set_title(
                f"Cluster size breakdown\n"
                f"{pct_clusters:.1f}% of clusters are singletons  |  "
                f"{pct_proteins:.1f}% of proteins are in singleton clusters"
            )
        else:
            ax.set_title(
                f"Cluster size breakdown\n"
                f"{pct_clusters:.1f}% of clusters are singletons"
            )
    else:
        ax.set_title("Cluster size breakdown")

    y_offset = 0.01 * max(counts, default=1)
    for bar, v in zip(bars, counts):
        if v > 0:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + y_offset,
                    str(v), ha="center", va="bottom", fontsize=8)


def plot_cluster_length_variation(results_root: str, ax: Any) -> None:
    """Scatter of within-cluster length std vs cluster size (multi-member clusters only)."""
    df = _safe_read(Path(results_root) / "01_mmseqs" / "subfamily_stats.tsv")
    if df is None or "std_length_aa" not in df.columns or "n_members" not in df.columns:
        ax.set_visible(False)
        return
    multi = df[df["n_members"] >= 2].copy()
    if multi.empty:
        ax.set_visible(False)
        return
    ax.scatter(
        multi["n_members"], multi["std_length_aa"],
        alpha=0.4, s=10, color="#4c72b0", rasterized=True,
    )
    ax.set_xlabel("Cluster size (members)")
    ax.set_ylabel("Std dev of sequence length (aa)")
    ax.set_title("Within-cluster length variation")
    ax.set_xscale("log")


def plot_cluster_identity_range(results_root: str, ax: Any) -> None:
    """Scatter of within-cluster pident spread (max\u2212min) vs cluster size."""
    df = _safe_read(Path(results_root) / "01_mmseqs" / "subfamily_stats.tsv")
    if (
        df is None
        or "min_pident" not in df.columns
        or "max_pident" not in df.columns
        or "n_members" not in df.columns
    ):
        ax.set_visible(False)
        return
    multi = df[df["n_members"] >= 2].copy()
    multi = multi.dropna(subset=["min_pident", "max_pident"])
    if multi.empty:
        ax.set_visible(False)
        return
    pident_spread = multi["max_pident"] - multi["min_pident"]
    ax.scatter(
        multi["n_members"], pident_spread,
        alpha=0.4, s=10, color="#c44e52", rasterized=True,
    )
    ax.set_xlabel("Cluster size (members)")
    ax.set_ylabel("Identity range (max \u2212 min pident, %)")
    ax.set_title("Within-cluster identity range")
    ax.set_xscale("log")


# ---------------------------------------------------------------------------
# MMseqs2 clustering step — publication-quality plots via plotnine
# ---------------------------------------------------------------------------

def generate_mmseqs_cluster_plots(
    outdir: str | Path,
    logger=None,
    resume: bool = False,
) -> Path | None:
    """Generate publication-quality QC plots for the MMseqs2 clustering step.

    Uses the **plotnine** library (install via ``pip install plotnine`` or
    ``pip install 'plm-cluster[plots]'``) to produce figure-ready PNG files
    saved to ``<outdir>/plots/``.

    Four plots are produced:

    * ``subfamily_size_distribution.png`` -- log-scale histogram of cluster
      member counts with median line.
    * ``cluster_size_breakdown.png`` -- bar chart of singleton / 2-member /
      >=3-member cluster counts with percentage annotations.
    * ``cluster_length_variation.png`` -- scatter of within-cluster sequence
      length standard deviation vs cluster size (multi-member clusters only).
    * ``cluster_identity_range.png`` -- scatter of within-cluster pident
      range (max - min) vs cluster size.

    Parameters
    ----------
    outdir : str or Path
        MMseqs2 clustering output directory (``results/01_mmseqs`` by default).
        Plots are written to ``<outdir>/plots/``.
    logger : logging.Logger, optional
        Logger for status messages.
    resume : bool, optional
        If *True* and plots already exist, skip generation.

    Returns
    -------
    Path or None
        Path to the plots sub-directory, or *None* if plotnine is unavailable
        or required data files are missing.
    """
    out = Path(outdir)
    plots_dir = out / "plots"
    sentinel = plots_dir / "subfamily_size_distribution.png"

    if resume and sentinel.exists():
        if logger:
            logger.info(
                "Resume: MMseqs cluster plots already exist in %s, skipping.", str(plots_dir)
            )
        return plots_dir

    try:
        from plotnine import (  # noqa: PLC0415
            aes,
            element_blank,
            element_text,
            geom_col,
            geom_histogram,
            geom_point,
            geom_text,
            geom_vline,
            ggplot,
            labs,
            scale_fill_manual,
            scale_x_log10,
            scale_y_log10,
            theme,
            theme_classic,
        )
    except ImportError:
        if logger:
            logger.warning(
                "plotnine not installed — skipping MMseqs cluster plots. "
                "Install with: pip install plotnine  or  pip install 'plm-cluster[plots]'"
            )
        return None

    stats_df = _safe_read(out / "subfamily_stats.tsv")
    if stats_df is None:
        if logger:
            logger.warning(
                "subfamily_stats.tsv not found in %s — skipping MMseqs cluster plots.", str(out)
            )
        return None

    plots_dir.mkdir(parents=True, exist_ok=True)

    _THEME = (
        theme_classic(base_size=12)
        + theme(
            plot_title=element_text(size=13, face="bold", margin={"b": 6}),
            axis_title=element_text(size=11),
            axis_text=element_text(size=10),
        )
    )
    _DPI = 300
    _W, _H = 8, 5  # inches

    saved: list[Path] = []

    # ------------------------------------------------------------------
    # 1. Subfamily size distribution (log-scale histogram)
    # ------------------------------------------------------------------
    median_size = float(stats_df["n_members"].median())
    try:
        p1 = (
            ggplot(stats_df, aes(x="n_members"))
            + geom_histogram(bins=50, fill="#4c72b0", color="white", size=0.3)
            + geom_vline(xintercept=median_size, color="#c44e52", linetype="dashed", size=0.8)
            + scale_y_log10()
            + labs(
                title="Subfamily size distribution",
                x="Members per subfamily",
                y="Count (log\u2081\u2080 scale)",
                caption=f"Dashed line: median = {median_size:.0f}",
            )
            + _THEME
        )
        path1 = plots_dir / "subfamily_size_distribution.png"
        p1.save(str(path1), dpi=_DPI, width=_W, height=_H, verbose=False)
        saved.append(path1)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("Could not generate subfamily_size_distribution plot: %s", exc)

    # ------------------------------------------------------------------
    # 2. Cluster size breakdown (bar chart — singletons / 2-member / >=3)
    # ------------------------------------------------------------------
    n_singletons = int((stats_df["n_members"] == 1).sum())
    n_2 = int((stats_df["n_members"] == 2).sum())
    n_3plus = int((stats_df["n_members"] >= 3).sum())
    n_total_clusters = n_singletons + n_2 + n_3plus
    n_proteins_total = int(stats_df["n_members"].sum())

    pct_clusters = 100.0 * n_singletons / n_total_clusters if n_total_clusters > 0 else 0.0
    pct_proteins = 100.0 * n_singletons / n_proteins_total if n_proteins_total > 0 else 0.0

    bar_df = pd.DataFrame({
        "Category": ["Singletons\n(n=1)", "2-member\nclusters", "\u22653-member\nclusters"],
        "Count": [n_singletons, n_2, n_3plus],
        "Fill": ["Singletons", "2-member", "\u22653-member"],
    })
    bar_df["Category"] = pd.Categorical(
        bar_df["Category"], categories=bar_df["Category"].tolist(), ordered=True
    )
    bar_df["label"] = bar_df["Count"].apply(str)
    subtitle = (
        f"{pct_clusters:.1f}% of clusters are singletons  |  "
        f"{pct_proteins:.1f}% of proteins are in singleton clusters"
    )
    try:
        p2 = (
            ggplot(bar_df, aes(x="Category", y="Count", fill="Fill"))
            + geom_col(color="white", show_legend=False)
            + geom_text(
                aes(label="label"),
                va="bottom",
                size=9,
                nudge_y=0.01 * max(int(bar_df["Count"].max()), 1),
            )
            + scale_fill_manual(
                values={"Singletons": "#dd8452", "2-member": "#4c72b0", "\u22653-member": "#55a868"}
            )
            + labs(
                title="Cluster size breakdown",
                subtitle=subtitle,
                x="",
                y="Cluster count",
            )
            + _THEME
            + theme(axis_line_x=element_blank(), axis_ticks_major_x=element_blank())
        )
        path2 = plots_dir / "cluster_size_breakdown.png"
        p2.save(str(path2), dpi=_DPI, width=_W, height=_H, verbose=False)
        saved.append(path2)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("Could not generate cluster_size_breakdown plot: %s", exc)

    # ------------------------------------------------------------------
    # 3. Within-cluster length variation (scatter, multi-member only)
    # ------------------------------------------------------------------
    if "std_length_aa" in stats_df.columns:
        multi = stats_df[stats_df["n_members"] >= 2].copy()
        if not multi.empty:
            try:
                p3 = (
                    ggplot(multi, aes(x="n_members", y="std_length_aa"))
                    + geom_point(alpha=0.35, size=1.5, color="#4c72b0")
                    + scale_x_log10()
                    + labs(
                        title="Within-cluster length variation",
                        x="Cluster size (members, log\u2081\u2080 scale)",
                        y="Std dev of sequence length (aa)",
                    )
                    + _THEME
                )
                path3 = plots_dir / "cluster_length_variation.png"
                p3.save(str(path3), dpi=_DPI, width=_W, height=_H, verbose=False)
                saved.append(path3)
            except Exception as exc:  # noqa: BLE001
                if logger:
                    logger.warning("Could not generate cluster_length_variation plot: %s", exc)

    # ------------------------------------------------------------------
    # 4. Within-cluster identity range (scatter, multi-member only)
    # ------------------------------------------------------------------
    if "min_pident" in stats_df.columns and "max_pident" in stats_df.columns:
        id_df = (
            stats_df[stats_df["n_members"] >= 2]
            .dropna(subset=["min_pident", "max_pident"])
            .copy()
        )
        if not id_df.empty:
            id_df["pident_spread"] = id_df["max_pident"] - id_df["min_pident"]
            try:
                p4 = (
                    ggplot(id_df, aes(x="n_members", y="pident_spread"))
                    + geom_point(alpha=0.35, size=1.5, color="#c44e52")
                    + scale_x_log10()
                    + labs(
                        title="Within-cluster identity range",
                        x="Cluster size (members, log\u2081\u2080 scale)",
                        y="Identity range (max \u2212 min pident, %)",
                    )
                    + _THEME
                )
                path4 = plots_dir / "cluster_identity_range.png"
                p4.save(str(path4), dpi=_DPI, width=_W, height=_H, verbose=False)
                saved.append(path4)
            except Exception as exc:  # noqa: BLE001
                if logger:
                    logger.warning("Could not generate cluster_identity_range plot: %s", exc)

    if logger:
        if saved:
            logger.info(
                "MMseqs cluster plots (%d files) saved to %s", len(saved), str(plots_dir)
            )
        else:
            logger.warning("No MMseqs cluster plots were produced (check data files).")
    return plots_dir if saved else None


# ---------------------------------------------------------------------------
# OrthoFinder subclustering — OG split/retained plots via plotnine
# ---------------------------------------------------------------------------

def generate_orthofinder_split_plots(
    outdir: str | Path,
    logger=None,
    resume: bool = False,
) -> "Path | None":
    """Generate publication-quality QC plots for the OrthoFinder MMSeqs2 subclustering step.

    Reads ``og_subfamily_map.tsv`` and ``subfamily_stats.tsv`` produced by
    :func:`~plm_cluster.pipeline.orthofinder_cluster` and writes five
    figure-ready PNG files (300 DPI, ``theme_classic``) to
    ``<outdir>/plots/``.

    Five plots are produced:

    * ``og_split_summary.png`` -- bar chart of split vs retained OG counts
      with percentage annotations.
    * ``subfamilies_per_split_og.png`` -- histogram of the number of
      subfamilies generated from OGs that were split into more than one
      subfamily.
    * ``og_size_vs_subfamilies.png`` -- scatter plot of original OG member
      count vs number of subfamilies produced (log–log scale), coloured by
      split status.
    * ``split_fraction_by_og_size.png`` -- stacked bar chart showing the
      fraction of OGs that were split or retained, broken down by OG size
      bin.
    * ``subfamily_count_from_splits.png`` -- bar chart of the total number
      of subfamilies contributed by split vs retained OGs, illustrating the
      downstream impact of splitting.

    Parameters
    ----------
    outdir : str or Path
        OrthoFinder clustering output directory (``results/01_orthofinder``
        by default).  Plots are written to ``<outdir>/plots/``.
    logger : logging.Logger, optional
        Logger for status messages.
    resume : bool, optional
        If *True* and plots already exist, skip generation.

    Returns
    -------
    Path or None
        Path to the plots sub-directory, or *None* if **plotnine** is not
        installed, if ``og_subfamily_map.tsv`` is absent, or if no plots
        could be saved (e.g. all rendering attempts raised exceptions).
    """
    out = Path(outdir)
    plots_dir = out / "plots"
    sentinel = plots_dir / "og_split_summary.png"

    if resume and sentinel.exists():
        if logger:
            logger.info(
                "Resume: OrthoFinder split plots already exist in %s, skipping.",
                str(plots_dir),
            )
        return plots_dir

    try:
        from plotnine import (  # noqa: PLC0415
            aes,
            element_blank,
            element_text,
            geom_col,
            geom_histogram,
            geom_point,
            geom_text,
            ggplot,
            labs,
            scale_color_manual,
            scale_fill_manual,
            scale_x_log10,
            scale_y_log10,
            theme,
            theme_classic,
        )
    except ImportError:
        if logger:
            logger.warning(
                "plotnine not installed — skipping OrthoFinder split plots. "
                "Install with: pip install plotnine  or  pip install 'plm-cluster[plots]'"
            )
        return None

    og_map_path = out / "og_subfamily_map.tsv"
    stats_path = out / "subfamily_stats.tsv"

    og_map = _safe_read(og_map_path)
    stats_df = _safe_read(stats_path)

    if og_map is None:
        if logger:
            logger.warning(
                "og_subfamily_map.tsv not found in %s — skipping OrthoFinder split plots.",
                str(out),
            )
        return None

    # ------------------------------------------------------------------
    # Build per-OG summary: n_subfamilies, total_members, was_split
    # ------------------------------------------------------------------
    og_subfam_counts = og_map.groupby("og_id")["subfamily_id"].nunique().reset_index()
    og_subfam_counts.columns = ["og_id", "n_subfamilies"]

    if stats_df is not None and "n_members" in stats_df.columns:
        # Merge member counts onto the og_map then sum per OG
        merged = og_map.merge(stats_df[["subfamily_id", "n_members"]], on="subfamily_id", how="left")
        og_sizes = merged.groupby("og_id")["n_members"].sum().reset_index()
        og_sizes.columns = ["og_id", "total_members"]
        og_df = og_subfam_counts.merge(og_sizes, on="og_id", how="left")
    else:
        og_df = og_subfam_counts.copy()
        og_df["total_members"] = np.nan

    og_df["was_split"] = og_df["n_subfamilies"] > 1
    og_df["split_label"] = og_df["was_split"].map({True: "Split", False: "Retained"})

    n_split = int(og_df["was_split"].sum())
    n_retained = int((~og_df["was_split"]).sum())
    n_total_ogs = n_split + n_retained
    pct_split = 100.0 * n_split / n_total_ogs if n_total_ogs > 0 else 0.0
    pct_retained = 100.0 * n_retained / n_total_ogs if n_total_ogs > 0 else 0.0

    plots_dir.mkdir(parents=True, exist_ok=True)

    _THEME = (
        theme_classic(base_size=12)
        + theme(
            plot_title=element_text(size=13, face="bold", margin={"b": 6}),
            axis_title=element_text(size=11),
            axis_text=element_text(size=10),
        )
    )
    _DPI = 300
    _W, _H = 8, 5

    saved: list[Path] = []

    # ------------------------------------------------------------------
    # 1. Split vs retained OG summary bar chart
    # ------------------------------------------------------------------
    summary_df = pd.DataFrame({
        "Status": ["Retained\n(1 subfamily)", "Split\n(>1 subfamily)"],
        "Count": [n_retained, n_split],
        "Fill": ["Retained", "Split"],
        "pct": [pct_retained, pct_split],
    })
    summary_df["Status"] = pd.Categorical(
        summary_df["Status"],
        categories=summary_df["Status"].tolist(),
        ordered=True,
    )
    summary_df["label"] = summary_df.apply(
        lambda r: f"{int(r['Count'])}\n({r['pct']:.1f}%)", axis=1
    )
    try:
        p1 = (
            ggplot(summary_df, aes(x="Status", y="Count", fill="Fill"))
            + geom_col(color="white", show_legend=False)
            + geom_text(
                aes(label="label"),
                va="bottom",
                size=9,
                nudge_y=0.01 * max(int(summary_df["Count"].max()), 1),
            )
            + scale_fill_manual(values={"Retained": "#4c72b0", "Split": "#c44e52"})
            + labs(
                title="Orthogroup split summary",
                subtitle=f"{n_total_ogs} OGs processed  |  "
                         f"{pct_split:.1f}% split into multiple subfamilies",
                x="",
                y="Number of orthogroups",
            )
            + _THEME
            + theme(axis_line_x=element_blank(), axis_ticks_major_x=element_blank())
        )
        path1 = plots_dir / "og_split_summary.png"
        p1.save(str(path1), dpi=_DPI, width=_W, height=_H, verbose=False)
        saved.append(path1)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("Could not generate og_split_summary plot: %s", exc)

    # ------------------------------------------------------------------
    # 2. Histogram of subfamilies per split OG
    # ------------------------------------------------------------------
    split_og_df = og_df[og_df["was_split"]].copy()
    if not split_og_df.empty:
        n_bins = min(50, max(10, len(split_og_df) // 5))
        median_subfams = float(split_og_df["n_subfamilies"].median())
        try:
            p2 = (
                ggplot(split_og_df, aes(x="n_subfamilies"))
                + geom_histogram(bins=n_bins, fill="#c44e52", color="white", size=0.3)
                + scale_y_log10()
                + labs(
                    title="Subfamilies generated from split orthogroups",
                    subtitle=f"{n_split} split OGs  |  median subfamilies per split OG = {median_subfams:.0f}",
                    x="Number of subfamilies",
                    y="Number of OGs (log\u2081\u2080 scale)",
                )
                + _THEME
            )
            path2 = plots_dir / "subfamilies_per_split_og.png"
            p2.save(str(path2), dpi=_DPI, width=_W, height=_H, verbose=False)
            saved.append(path2)
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.warning("Could not generate subfamilies_per_split_og plot: %s", exc)

    # ------------------------------------------------------------------
    # 3. OG size vs number of subfamilies scatter (log–log)
    # ------------------------------------------------------------------
    scatter_df = og_df.dropna(subset=["total_members"]).copy()
    scatter_df = scatter_df[scatter_df["total_members"] > 0]
    if not scatter_df.empty:
        try:
            p3 = (
                ggplot(scatter_df, aes(x="total_members", y="n_subfamilies", color="split_label"))
                + geom_point(alpha=0.4, size=1.5)
                + scale_x_log10()
                + scale_y_log10()
                + scale_color_manual(values={"Retained": "#4c72b0", "Split": "#c44e52"})
                + labs(
                    title="Orthogroup size vs subfamilies produced",
                    x="OG member count (log\u2081\u2080 scale)",
                    y="Subfamilies produced (log\u2081\u2080 scale)",
                    color="Status",
                )
                + _THEME
            )
            path3 = plots_dir / "og_size_vs_subfamilies.png"
            p3.save(str(path3), dpi=_DPI, width=_W, height=_H, verbose=False)
            saved.append(path3)
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.warning("Could not generate og_size_vs_subfamilies plot: %s", exc)

    # ------------------------------------------------------------------
    # 4. Split fraction by OG size bin (stacked bar)
    # ------------------------------------------------------------------
    size_df = og_df.dropna(subset=["total_members"]).copy()
    if not size_df.empty:
        _bins = [0, 1, 5, 20, 100, 500, float("inf")]
        _labels = ["1", "2–5", "6–20", "21–100", "101–500", "501+"]
        size_df["size_bin"] = pd.cut(
            size_df["total_members"],
            bins=_bins,
            labels=_labels,
            right=True,
        )
        bin_counts = (
            size_df.groupby(["size_bin", "split_label"], observed=True)
            .size()
            .reset_index(name="Count")
        )
        bin_totals = bin_counts.groupby("size_bin", observed=True)["Count"].transform("sum")
        bin_counts["fraction"] = bin_counts["Count"] / bin_totals
        bin_counts["label"] = bin_counts["Count"].apply(str)
        bin_counts["size_bin"] = pd.Categorical(
            bin_counts["size_bin"].astype(str), categories=_labels, ordered=True
        )
        try:
            p4 = (
                ggplot(bin_counts, aes(x="size_bin", y="Count", fill="split_label"))
                + geom_col(color="white", position="stack")
                + scale_fill_manual(values={"Retained": "#4c72b0", "Split": "#c44e52"})
                + labs(
                    title="OG split status by orthogroup size",
                    subtitle="Larger OGs tend to be split into multiple subfamilies",
                    x="OG member count (bin)",
                    y="Number of OGs",
                    fill="Status",
                )
                + _THEME
            )
            path4 = plots_dir / "split_fraction_by_og_size.png"
            p4.save(str(path4), dpi=_DPI, width=_W, height=_H, verbose=False)
            saved.append(path4)
        except Exception as exc:  # noqa: BLE001
            if logger:
                logger.warning("Could not generate split_fraction_by_og_size plot: %s", exc)

    # ------------------------------------------------------------------
    # 5. Total subfamilies contributed by split vs retained OGs
    # ------------------------------------------------------------------
    contrib_df = (
        og_df.groupby("split_label")["n_subfamilies"]
        .sum()
        .reset_index()
        .rename(columns={"n_subfamilies": "total_subfamilies"})
    )
    contrib_df["label"] = contrib_df["total_subfamilies"].apply(str)
    # Ensure consistent ordering
    contrib_df["split_label"] = pd.Categorical(
        contrib_df["split_label"], categories=["Retained", "Split"], ordered=True
    )
    contrib_df = contrib_df.sort_values("split_label")
    try:
        p5 = (
            ggplot(contrib_df, aes(x="split_label", y="total_subfamilies", fill="split_label"))
            + geom_col(color="white", show_legend=False)
            + geom_text(
                aes(label="label"),
                va="bottom",
                size=9,
                nudge_y=0.01 * max(int(contrib_df["total_subfamilies"].max()), 1),
            )
            + scale_fill_manual(values={"Retained": "#4c72b0", "Split": "#c44e52"})
            + labs(
                title="Total subfamilies contributed by split vs retained OGs",
                subtitle="Split OGs generate more subfamilies, expanding coverage",
                x="OG status",
                y="Total subfamilies",
            )
            + _THEME
            + theme(axis_line_x=element_blank(), axis_ticks_major_x=element_blank())
        )
        path5 = plots_dir / "subfamily_count_from_splits.png"
        p5.save(str(path5), dpi=_DPI, width=_W, height=_H, verbose=False)
        saved.append(path5)
    except Exception as exc:  # noqa: BLE001
        if logger:
            logger.warning("Could not generate subfamily_count_from_splits plot: %s", exc)

    if logger:
        if saved:
            logger.info(
                "OrthoFinder split plots (%d files) saved to %s", len(saved), str(plots_dir)
            )
        else:
            logger.warning("No OrthoFinder split plots were produced (check data files).")
    return plots_dir if saved else None


# ---------------------------------------------------------------------------
# Summary dashboard (full pipeline QC, matplotlib)
# ---------------------------------------------------------------------------

def generate_qc_plots(results_root: str, logger=None, resume: bool = False) -> Path | None:
    """Generate a multi-panel QC summary figure.

    Parameters
    ----------
    results_root : str
        Path to the pipeline results directory (e.g. ``results/``).
    logger : logging.Logger, optional
        Logger for status messages.
    resume : bool, optional
        If True and the summary figure already exists, skip generation.

    Returns
    -------
    Path or None
        Path to the saved summary figure, or None if matplotlib is unavailable.
    """
    out = Path(results_root) / "qc_plots"
    summary_path = out / "pipeline_summary.png"
    if resume and summary_path.exists():
        if logger:
            logger.info("Resume: QC plots already exist at %s, skipping.", str(out))
        return summary_path

    try:
        import matplotlib  # noqa: PLC0415
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # noqa: PLC0415
    except ImportError:
        if logger:
            logger.warning("matplotlib not installed — skipping QC plots")
        return None

    out.mkdir(parents=True, exist_ok=True)

    # 3x3 grid: original 6 plots + 3 new singleton/QC plots
    fig, axes = plt.subplots(3, 3, figsize=(21, 15))
    fig.suptitle("plm-cluster QC Summary", fontsize=16, fontweight="bold")

    plot_subfamily_sizes(results_root, axes[0, 0])
    plot_hmm_edge_weights(results_root, axes[0, 1])
    plot_embedding_cosines(results_root, axes[0, 2])
    plot_family_sizes(results_root, axes[1, 0])
    plot_protein_coverage(results_root, axes[1, 1])
    plot_fusion_fraction(results_root, axes[1, 2])
    plot_singleton_summary(results_root, axes[2, 0])
    plot_cluster_length_variation(results_root, axes[2, 1])
    plot_cluster_identity_range(results_root, axes[2, 2])

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(summary_path, dpi=150)
    plt.close(fig)

    # Also save individual plots
    plot_funcs = [
        ("subfamily_size_distribution.png", plot_subfamily_sizes),
        ("hmm_edge_weights.png", plot_hmm_edge_weights),
        ("embedding_cosine_distribution.png", plot_embedding_cosines),
        ("family_size_distribution.png", plot_family_sizes),
        ("protein_coverage_fraction.png", plot_protein_coverage),
        ("fusion_summary.png", plot_fusion_fraction),
        ("singleton_summary.png", plot_singleton_summary),
        ("cluster_length_variation.png", plot_cluster_length_variation),
        ("cluster_identity_range.png", plot_cluster_identity_range),
    ]
    for fname, func in plot_funcs:
        fig_i, ax_i = plt.subplots(figsize=(8, 5))
        func(results_root, ax_i)
        fig_i.tight_layout()
        fig_i.savefig(out / fname, dpi=150)
        plt.close(fig_i)

    if logger:
        logger.info("QC plots saved to %s", out)
    return summary_path
