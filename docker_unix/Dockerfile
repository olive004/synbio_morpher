# syntax=docker/dockerfile:1

# FROM intarna_docker
# https://quay.io/repository/biocontainers/intarna
# FROM nvidia/cuda:11.3.0-base-ubuntu20.04
FROM nvidia/cuda:11.6.2-base-ubuntu20.04

################################################################################
#  Install python dependencies
################################################################################
# RUN nvidia-smi
WORKDIR /usr/lib
COPY ./requirements.txt /usr/lib/install_requirements/requirements.txt
RUN apt-get update && \
    yes | apt-get install python3-pip
RUN cd /usr/lib/install_requirements && \
    pip install -r ./requirements.txt 
RUN pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html

################################################################################
#  setup environment
################################################################################
ENV ROOT_DIR=/workdir
ENV PYTHONPATH=/workdir
ENV NVIDIA_VISIBLE_DEVICES all
ENV NVIDIA_DRIVER_CAPABILITIES compute,utility
ENV MNT ./..
WORKDIR /workdir
# ENTRYPOINT ["/bin/bash"]

## A few missing files for dependencies:
# For conda: these 2 locations were determined by running the command 
# find /usr -name libz.so.1 
# on the container and on an Ubuntu machine. The first location is the one from 
# the container and the second location is where conda will look for libz.so.1
# RUN cp /usr/local/lib/libz.so.1 /usr/lib/x86_64-linux-gnu
# For jax: similar story
# RUN cp /usr/local/lib/libstdc++.so.6 /usr/local/lib/python3.10/site-packages/jax/_src/lib/

################################################################################
#  Install Conda & IntaRNA dependencies
################################################################################
# x RUN yes | apt-get install wget && \
#     wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# RUN curl https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
# x RUN (sleep 1; echo yes; sleep 1; echo /root/miniconda3; sleep 1; echo yes) | bash Miniconda3-latest-Linux-x86_64.sh 
    # export LD_LIBRARY_PATH=/root/miniconda3/lib/ && \
    # mkdir -p $CONDA_PREFIX/etc/conda/activate.d && \
    # mkdir -p $CONDA_PREFIX/etc/conda/activate.d && \
    # echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh && \
    # . /root/miniconda3/bin/activate && \
    # # yes | conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 && \
    # yes | conda install -c conda-forge -c bioconda intarna
# RUN export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
# RUN export LD_LIBRARY_PATH=/root/miniconda3/lib/
# RUN mkdir -p $CONDA_PREFIX/etc/conda/activate.d
# RUN echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
# RUN /root/miniconda3/bin/activate && yes | conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 && conda install -c conda-forge -c bioconda intarna
# x RUN . /root/miniconda3/bin/activate && \
#     yes | conda install -c conda-forge -c bioconda intarna

# RUN source /root/miniconda3/bin/activate && yes | conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0 && conda install -c conda-forge -c bioconda intarna
# RUN yes | conda install -c conda-forge -c bioconda intarna
# RUN conda deactivate

# Go
# RUN wget https://go.dev/dl/go1.19.4.linux-amd64.tar.gz
# RUN tar -C /usr/local -xzf go1.19.4.linux-amd64.tar.gz
# RUN export PATH=$PATH:/usr/local/go/bin
# RUN export GOPATH="/mnt/c/Users/${user}/directory/to/your/golang/workspace"


################################################################################
#  Misc patches
################################################################################
# For NVIDIA: find this path via ```find /usr/ -name libcudart.so.11.0```
# RUN export LD_LIBRARY_PATH=/usr/local/cuda-11.3/targets/x86_64-linux/lib/:/usr/local/cuda-11.3/compat/
