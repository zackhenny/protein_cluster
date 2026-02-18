from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="esm-extract",
        description="Compatibility wrapper for environments where fair-esm does not install an esm-extract binary.",
    )
    parser.add_argument("model_path", help="Path to local ESM model weights (.pt)")
    parser.add_argument("fasta", help="Input FASTA file")
    parser.add_argument("outdir", help="Output directory")
    parser.add_argument("--repr_layers", default="-1", help="Comma-separated repr layers (default: -1 = last)")
    parser.add_argument("--include", default="mean", help="Embeddings to include (default: mean)")
    parser.add_argument("--batch_size", type=int, default=4, help="Batch size for processing (default: 4)")
    args = parser.parse_args()

    try:
        import esm
        import torch
    except Exception as exc:  # pragma: no cover
        raise SystemExit(f"ERROR: esm-extract wrapper requires fair-esm + torch: {exc}")

    from .io_utils import read_fasta

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    model, alphabet = esm.pretrained.load_model_and_alphabet_local(args.model_path)
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    recs = list(read_fasta(args.fasta))
    layers = [int(x) for x in args.repr_layers.split(",") if x.strip()]
    if layers == [-1]:
        layers = [model.num_layers]

    data = [(r.id, r.seq) for r in recs]
    batch_size = args.batch_size
    total_batches = (len(data) + batch_size - 1) // batch_size
    print(f"Processing {len(data)} sequences in {total_batches} batches of size {batch_size}")
    
    for batch_idx in range(0, len(data), batch_size):
        batch_data = data[batch_idx:batch_idx + batch_size]
        batch_recs = recs[batch_idx:batch_idx + batch_size]
        batch_num = batch_idx // batch_size + 1
        print(f"Processing batch {batch_num}/{total_batches} ({len(batch_data)} sequences)")
        
        _, seqs, toks = batch_converter(batch_data)
        with torch.no_grad():
            out = model(toks, repr_layers=layers, return_contacts=False)

        reps = out["representations"][layers[-1]]
        for i, rec in enumerate(batch_recs):
            L = len(seqs[i])
            mean_vec = reps[i, 1 : L + 1].mean(0).cpu()
            torch.save({"mean_representations": {layers[-1]: mean_vec}}, outdir / f"{rec.id}.pt")
        
        # Clear GPU memory after each batch
        del toks, out, reps
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
