# Installation, environment setup, and container builds

This document covers local installation, conda setup, and container builds.

## Requirements

### External binaries (required at runtime)
- `mmseqs`
- `hhmake`
- `hhalign` (for `pairwise` HMM-HMM mode, the default)
- `mafft`

### Optional binaries
- `hhsearch` and `ffindex_build` (for `db-search` HMM-HMM mode)
- `mcl` (if you request MCL mode)

### Python requirements
- PyTorch
- fair-esm
- numpy/scipy/pandas
- scikit-learn (FAISS optional)
- python-igraph + leidenalg

### GPU support

Two pipeline stages benefit from GPU acceleration:

1. **Embedding** (`plm_cluster embed`): Set `embed.device: cuda` in your config.
   Requires PyTorch with CUDA support (installed via conda's `pytorch` channel or
   `pip install torch` with the appropriate CUDA index).

2. **KNN / rKCNN** (`plm_cluster knn`): Set `knn.device: cuda` in your config.
   Requires the `faiss-gpu` package:
   ```bash
   # conda (recommended for HPC)
   conda install -c conda-forge faiss-gpu
   # pip
   pip install faiss-gpu
   ```
   If `faiss-gpu` is not available, the pipeline falls back to CPU FAISS or sklearn.
   When `knn.mode: rkcnn`, PyTorch CUDA is also used for batch distance
   computations in random subspaces.

## Conda installation

```bash
conda env create -f envs/plm_cluster.yaml
# recommended on HPC (always installs into named env):
bash scripts/install_in_conda_env.sh plm_cluster . test,embed

# manual equivalent:
# conda activate plm_cluster
# python -m pip install --no-build-isolation --no-user .
```


### Why this avoids home-directory installs

- `conda run -n <env> python -m pip ...` binds `pip` to the target env interpreter.
- `--no-user` disables user-site installs.
- `PYTHONNOUSERSITE=1` prevents reading user site packages.

If you need to verify the target location:

```bash
conda run -n plm_cluster python -c "import sys; print(sys.executable); print(sys.prefix)"
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
# Get subcommand-specific help:
apptainer exec plm_cpu_graph.sif plm_cluster hmm-hmm-edges --help
```

## Single-container Docker build

```bash
docker build -t plm-cluster:latest -f container/Dockerfile .
```

Run example:

```bash
docker run --rm -it -v $PWD:/work -w /work plm-cluster:latest
```

> **Note:** The Docker entry point launches the top-level `plm_cluster` CLI.
> To see options for a specific subcommand, pass the subcommand and `--help`:
>
> ```bash
> docker run --rm plm-cluster:latest plm_cluster hmm-hmm-edges --help
> docker run --rm plm-cluster:latest plm_cluster run-all --help
> ```

## Reproducibility notes

- Each run writes `results/manifests/run_manifest.json` with parameters, tool paths, versions, git hash, and input checksums.
- External commands are logged in `results/logs/`.
- Keep config YAML checked into your project for exact reruns.
- The `hmm-hmm-edges` stage writes a real-time NDJSON progress file
  (`results/03_hmm_hmm_edges/hmm_hmm_progress.ndjson`, or
  `hmm_hmm_progress.shard_N.ndjson` for sharded runs) that enables safe
  resumption with `--resume`.  See
  [cli_workflow_and_options.md](cli_workflow_and_options.md) for details.

## Why `esm-extract` may be missing after installing `fair-esm`

Some `fair-esm` builds expose Python APIs but do not always install an `esm-extract` console binary into `bin/` (packaging/version dependent).

This repository now installs a compatibility wrapper command named `esm-extract` via `plm_cluster` itself.

Verify:

```bash
which esm-extract
esm-extract -h
```


## Installed test data

The package now includes a bundled toy FASTA at `plm_cluster/data/toy_proteins.faa`, so isolated environments running `pytest` can access test data without relying on repository-relative paths.


Pytest note: some HPC images miss `iniconfig` and `pygments`; the installer script now installs both into the conda env before running tests.
