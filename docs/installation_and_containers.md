# Installation, environment setup, and container builds

This document covers local installation, conda setup, and container builds.

## Requirements

### External binaries (required at runtime)
- `mmseqs`
- `hhmake`
- `hhalign` (or `hhsearch` if you adapt the HMM edge stage)

### Optional binaries
- `mcl` (if you request MCL mode)

### Python requirements
- PyTorch
- fair-esm
- numpy/scipy/pandas
- scikit-learn (FAISS optional)
- python-igraph + leidenalg

## Conda installation

```bash
conda env create -f envs/plm_cluster.yaml
conda activate plm_cluster
pip install .
# If your environment blocks build-isolation downloads:
# pip install . --no-build-isolation
```

Check CLI:

```bash
plm_cluster --help
```

## ESM offline weights

Embedding runs in offline mode and requires explicit local weights:

```bash
plm_cluster embed --reps_fasta results/01_mmseqs/subfamily_reps.faa --weights_path /path/to/esm2.pt
```

No silent model download is performed.

## Apptainer/Singularity build

Two-container pattern is recommended:

- CPU graph/profile image: `container/container_cpu_graph.def`
- GPU embedder image: `container/container_embedder.def`

Build commands:

```bash
apptainer build plm_cpu_graph.sif container/container_cpu_graph.def
apptainer build plm_embedder.sif container/container_embedder.def
```

Run example:

```bash
apptainer exec plm_cpu_graph.sif plm_cluster --help
apptainer exec --nv plm_embedder.sif plm_cluster --help
```

## Single-container Docker build

```bash
docker build -t plm-cluster:latest -f container/Dockerfile .
```

Run example:

```bash
docker run --rm -it -v $PWD:/work -w /work plm-cluster:latest
```

## Reproducibility notes

- Each run writes `results/manifests/run_manifest.json` with parameters, tool paths, versions, git hash, and input checksums.
- External commands are logged in `results/logs/`.
- Keep config YAML checked into your project for exact reruns.

## Why `esm-extract` may be missing after installing `fair-esm`

Some `fair-esm` builds expose Python APIs but do not always install an `esm-extract` console binary into `bin/` (packaging/version dependent).

This repository now installs a compatibility wrapper command named `esm-extract` via `plm_cluster` itself.

Verify:

```bash
which esm-extract
esm-extract -h
```

