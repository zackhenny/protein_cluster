#!/usr/bin/env python3
"""Generate sample OrthoFinder split plots from synthetic benchmark data.

Run from the repository root::

    python benchmark/generate_benchmark_plots.py

Output PNGs are written to ``benchmark/plots/``.

The synthetic dataset mimics a realistic OrthoFinder result where:
- Many small OGs (1–5 proteins) remain as single subfamilies.
- Medium OGs (6–100 proteins) are split with low-to-moderate probability.
- Large OGs (>100 proteins) are almost always split into several subfamilies.
"""
from __future__ import annotations

import os
import sys
import random
import tempfile
import csv
from pathlib import Path

# Allow running without installing the package (add src/ to path)
_REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_REPO_ROOT / "src"))

random.seed(42)


def _generate_synthetic_data(tmp_dir: Path) -> None:
    """Write og_subfamily_map.tsv and subfamily_stats.tsv to *tmp_dir*."""

    og_subfamily_rows: list[dict] = []
    stats_rows: list[dict] = []

    subfam_counter = 0

    def _next_subfam(og_id: str) -> str:
        nonlocal subfam_counter
        sfid = f"{og_id}_subfam_{subfam_counter:06d}"
        subfam_counter += 1
        return sfid

    # ----------------------------------------------------------------
    # Build a synthetic collection of OGs with realistic size distribution
    # ----------------------------------------------------------------
    og_specs: list[tuple[str, int]] = []  # (og_id, og_size)

    # 300 tiny OGs (size 1)
    for i in range(300):
        og_specs.append((f"OG{i:07d}", 1))

    # 400 small OGs (size 2–5)
    for i in range(300, 700):
        og_specs.append((f"OG{i:07d}", random.randint(2, 5)))

    # 250 medium OGs (size 6–30)
    for i in range(700, 950):
        og_specs.append((f"OG{i:07d}", random.randint(6, 30)))

    # 120 large OGs (size 31–200)
    for i in range(950, 1070):
        og_specs.append((f"OG{i:07d}", random.randint(31, 200)))

    # 50 very large OGs (size 201–1000)
    for i in range(1070, 1120):
        og_specs.append((f"OG{i:07d}", random.randint(201, 1000)))

    # 10 giant OGs (size 1001–5000)
    for i in range(1120, 1130):
        og_specs.append((f"OG{i:07d}", random.randint(1001, 5000)))

    # ----------------------------------------------------------------
    # Simulate subclustering outcomes
    # ----------------------------------------------------------------
    for og_id, og_size in og_specs:
        # Determine number of subfamilies produced based on OG size
        if og_size == 1:
            n_subfamilies = 1
        elif og_size <= 5:
            # 15% chance of split
            n_subfamilies = random.choices([1, 2, 3], weights=[85, 10, 5], k=1)[0]
        elif og_size <= 30:
            # 50% chance of split, up to 5 subfamilies
            max_split = max(2, og_size // 5)
            n_subfamilies = random.randint(1, min(max_split, 6))
        elif og_size <= 200:
            # Very likely to split
            max_split = max(2, og_size // 10)
            n_subfamilies = random.randint(2, min(max_split, 20))
        elif og_size <= 1000:
            max_split = max(3, og_size // 20)
            n_subfamilies = random.randint(3, min(max_split, 50))
        else:
            max_split = max(5, og_size // 50)
            n_subfamilies = random.randint(5, min(max_split, 100))

        # Assign proteins to subfamilies
        members_per_subfam = [og_size // n_subfamilies] * n_subfamilies
        for j in range(og_size % n_subfamilies):
            members_per_subfam[j] += 1

        for mem_count in members_per_subfam:
            sfid = _next_subfam(og_id)
            og_subfamily_rows.append({"og_id": og_id, "subfamily_id": sfid})
            stats_rows.append({
                "subfamily_id": sfid,
                "n_members": mem_count,
                "rep_protein_id": f"{sfid}_p0",
                "rep_length_aa": random.randint(100, 800),
                "is_singleton": 1 if mem_count == 1 else 0,
                "min_length_aa": random.randint(80, 200),
                "max_length_aa": random.randint(300, 900),
                "mean_length_aa": random.uniform(200, 600),
                "std_length_aa": random.uniform(0, 100),
                "min_pident": "",
                "max_pident": "",
                "mean_pident": "",
            })

    # ----------------------------------------------------------------
    # Write TSV files
    # ----------------------------------------------------------------
    og_map_path = tmp_dir / "og_subfamily_map.tsv"
    with og_map_path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["og_id", "subfamily_id"], delimiter="\t")
        writer.writeheader()
        writer.writerows(og_subfamily_rows)

    stats_path = tmp_dir / "subfamily_stats.tsv"
    with stats_path.open("w", newline="") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "subfamily_id", "n_members", "rep_protein_id", "rep_length_aa",
                "is_singleton", "min_length_aa", "max_length_aa", "mean_length_aa",
                "std_length_aa", "min_pident", "max_pident", "mean_pident",
            ],
            delimiter="\t",
        )
        writer.writeheader()
        writer.writerows(stats_rows)

    print(f"Synthetic data written to {tmp_dir}")


def main() -> None:
    benchmark_dir = Path(__file__).parent
    plots_dir = benchmark_dir / "plots"

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        _generate_synthetic_data(tmp_path)

        # Symlink or copy plots output to benchmark/plots/
        from plm_cluster.qc_plots import generate_orthofinder_split_plots  # noqa: PLC0415

        import logging
        logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
        logger = logging.getLogger("benchmark")

        result = generate_orthofinder_split_plots(tmp_path, logger=logger)

        if result is None:
            print(
                "\nERROR: No plots were produced. "
                "Install plotnine: pip install plotnine pandas",
                file=sys.stderr,
            )
            sys.exit(1)

        # Move/copy generated plots into benchmark/plots/
        import shutil
        if plots_dir.exists():
            shutil.rmtree(plots_dir)
        shutil.copytree(result, plots_dir)

    print(f"\nSample plots saved to: {plots_dir}")
    for p in sorted(plots_dir.iterdir()):
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
