# FIRST: build the image with:
#   docker build -t ser_project-env:latest .
# SECOND: run (powershell) the container with:
#   docker run -it --rm -v "$($PWD.Path):/workspace" -w /workspace ser_project-env:latest
# THIRD: type 'exit' to end the container session, or
#   'Ctrl+P' then 'Ctrl+Q' to detach and leave it running in the background

FROM mambaorg/micromamba:1.5.0
WORKDIR /workspace

# ensure we have root permissions for apt operations
USER root

# install system build deps for any remaining pip packages
RUN apt-get update && apt-get install -y build-essential cmake libsndfile1-dev libsoxr-dev pkg-config && rm -rf /var/lib/apt/lists/*

# create conda env with python + prebuilt numba/llvmlite from conda-forge
RUN micromamba create -y -n ser python=3.10 numba llvmlite -c conda-forge && micromamba clean -a -y

ENV PATH=/opt/conda/envs/ser/bin:$PATH
ENV MAMBA_ROOT_PREFIX=/opt/conda

# copy requirements first to leverage Docker layer caching
COPY SER_Project/requirements.txt /workspace/requirements.txt

# install Python deps into the micromamba env at build time
RUN /opt/conda/envs/ser/bin/python -m pip install -r /workspace/requirements.txt

# copy the rest of the workspace
COPY SER_Project/ /workspace/

CMD ["/bin/bash"]