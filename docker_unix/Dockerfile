# syntax=docker/dockerfile:1

# FROM intarna_docker
# https://quay.io/repository/biocontainers/intarna
FROM quay.io/biocontainers/intarna:3.3.1--pl5321h7ff8a90_0

################################################################################
#  Install python dependencies
################################################################################
WORKDIR /usr/lib
RUN pwd
RUN ls /
RUN ls /mnt
RUN ls /usr 
RUN pwd
RUN ls -R docker_unix
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
# ENTRYPOINT ["/bin/bash"]


