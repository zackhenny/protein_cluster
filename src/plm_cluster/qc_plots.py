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

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("plm-cluster QC Summary", fontsize=16, fontweight="bold")

    plot_subfamily_sizes(results_root, axes[0, 0])
    plot_hmm_edge_weights(results_root, axes[0, 1])
    plot_embedding_cosines(results_root, axes[0, 2])
    plot_family_sizes(results_root, axes[1, 0])
    plot_protein_coverage(results_root, axes[1, 1])
    plot_fusion_fraction(results_root, axes[1, 2])

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
