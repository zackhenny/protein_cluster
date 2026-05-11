"""QC plots for plm-cluster pipeline outputs.

Generates diagnostic figures from pipeline intermediate files and saves them
to ``results/qc_plots/``.  Requires matplotlib; gracefully skips if missing.
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
# Individual plot functions
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

    categories = ["Singletons\n(n=1)", "2-member\nclusters", "≥3-member\nclusters"]
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
    """Scatter of within-cluster pident spread (max−min) vs cluster size."""
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
    ax.set_ylabel("Identity range (max − min pident, %)")
    ax.set_title("Within-cluster identity range")
    ax.set_xscale("log")


# ---------------------------------------------------------------------------
# Summary dashboard
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
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        if logger:
            logger.warning("matplotlib not installed — skipping QC plots")
        return None

    out.mkdir(parents=True, exist_ok=True)

    # 3×3 grid: original 6 plots + 3 new singleton/QC plots
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
