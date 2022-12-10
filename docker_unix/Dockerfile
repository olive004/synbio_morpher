# syntax=docker/dockerfile:1

# FROM ubuntu
# RUN apt-get install libstdc++6

# FROM intarna_docker
# https://quay.io/repository/biocontainers/intarna
FROM quay.io/biocontainers/intarna:3.3.2--pl5321h7ff8a90_0

# COPY --from=0 /usr /

################################################################################
#  Install python dependencies
################################################################################
ENV DEBIAN_FRONTEND=noninteractive
# RUN apt update && apt install -y libstdc++6
WORKDIR /usr/lib
COPY ./requirements.txt /usr/lib/install_requirements/requirements.txt
RUN cd /usr/lib/install_requirements && \
    pip install -r ./requirements.txt 

################################################################################
#  setup environment
################################################################################
ENV ROOT_DIR=/workdir
ENV PYTHONPATH=/workdir
ENV MNT ./..
WORKDIR /workdir
COPY /usr/local/lib/libstdc++.so.6 /usr/local/lib/python3.10/site-packages/jax/_src/lib/
# ENTRYPOINT ["/bin/bash"]

# GPU
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -o Miniconda3-latest-Linux-x86_64.sh
RUN bash Miniconda3-latest-Linux-x86_64.sh
RUN conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
RUN mkdir -p $CONDA_PREFIX/etc/conda/activate.d
RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh