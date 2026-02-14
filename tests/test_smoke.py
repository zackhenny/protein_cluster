from pathlib import Path

import pandas as pd

from plm_cluster.config import load_config
from plm_cluster.pipeline import map_proteins_to_families


def test_smoke_mapping(tmp_path: Path):
    fasta = tmp_path / "toy.faa"
    fasta.write_text(">p1\nAAAAAA\n>p2\nBBBBBB\n")
    sf2f = tmp_path / "sf2f.tsv"
    pd.DataFrame([{"subfamily_id": "s1", "family_id": "f1"}]).to_csv(sf2f, sep="\t", index=False)
    smap = tmp_path / "smap.tsv"
    pd.DataFrame([
        {"protein_id": "p1", "subfamily_id": "s1", "is_rep": 1},
        {"protein_id": "p2", "subfamily_id": "s1", "is_rep": 0},
    ]).to_csv(smap, sep="\t", index=False)
    out = tmp_path / "05_domain_hits"
    map_proteins_to_families(str(fasta), str(sf2f), str(smap), str(out), load_config(None))
    assert (out / "protein_architectures.tsv").exists()
