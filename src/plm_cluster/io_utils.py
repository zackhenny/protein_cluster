from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


@dataclass(frozen=True)
class FastaRecord:
    id: str
    seq: str


def read_fasta(path: str | Path) -> list[FastaRecord]:
    records: list[FastaRecord] = []
    rec_id: str | None = None
    seq_parts: list[str] = []
    with Path(path).open() as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            if line.startswith(">"):
                if rec_id is not None:
                    records.append(FastaRecord(rec_id, "".join(seq_parts)))
                rec_id = line[1:].split()[0]
                seq_parts = []
            else:
                seq_parts.append(line)
    if rec_id is not None:
        records.append(FastaRecord(rec_id, "".join(seq_parts)))
    return records


def write_fasta(path: str | Path, records: Iterable[FastaRecord]) -> None:
    with Path(path).open("w") as handle:
        for rec in records:
            handle.write(f">{rec.id}\n")
            handle.write(f"{rec.seq}\n")
