FROM mambaorg/micromamba:1.5.8
COPY envs/plm_cluster.yaml /tmp/env.yaml
RUN micromamba create -y -f /tmp/env.yaml -n plm_cluster && micromamba clean -a -y
SHELL ["/bin/bash", "-lc"]
ENV MAMBA_DOCKERFILE_ACTIVATE=1
ENV CONDA_DEFAULT_ENV=plm_cluster
COPY . /opt/plm_cluster
WORKDIR /opt/plm_cluster
RUN source /opt/conda/etc/profile.d/conda.sh && conda activate plm_cluster && pip install -e .
ENTRYPOINT ["/bin/bash", "-lc", "source /opt/conda/etc/profile.d/conda.sh && conda activate plm_cluster && plm_cluster --help"]
