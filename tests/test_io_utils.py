from pathlib import Path

from plm_cluster.io_utils import read_fasta


def test_read_fasta_removes_stop_codons(tmp_path: Path):
    fasta = tmp_path / "stop_codons.faa"
    fasta.write_text(">p1\nMKT*AY*\n>p2\n*GAVLILKK*\n")

    records = read_fasta(fasta)

    assert [r.id for r in records] == ["p1", "p2"]
    assert [r.seq for r in records] == ["MKTAY", "GAVLILKK"]
