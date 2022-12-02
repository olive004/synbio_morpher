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
# RUN apt-get install libstdc++6
# RUN ubuntu bash -c "apt install libstdc"

################################################################################
#  setup environment
################################################################################
ENV ROOT_DIR=/workdir
ENV PYTHONPATH=/workdir
ENV MNT ./..
WORKDIR /workdir
COPY /usr/local/lib/libstdc++.so.6 /usr/local/lib/python3.10/site-packages/jax/_src/lib/
# ENTRYPOINT ["/bin/bash"]