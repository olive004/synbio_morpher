# syntax=docker/dockerfile:1

# FROM intarna_docker
# quay.io/repository/biocontainers/intarna
# FROM 16a5962af907:3.2.0--h176a8bc_0
FROM quay.io/biocontainers/intarna:3.2.0--h176a8bc_0
# WORKDIR /Users/oliviagallup/Desktop/Kode/Oxford/DPhil/Gene_circuit_glitch_prediction

################################################################################
#  Install python dependencies
################################################################################
WORKDIR /usr/lib
# COPY ./requirements.txt /usr/lib/install_requirements/requirements.txt
# RUN cd /usr/lib/install_requirements && \
#     pip install -r ./requirements.txt

################################################################################
#  setup environment
################################################################################
ENV ROOT_DIR=/workdir
# ENV TORCH_HOME=/workdir/src/pkg/model_zoo
ENV PYTHONPATH=/workdir
ENV MNT ../
WORKDIR /workdir
# ENTRYPOINT ["/bin/bash"]
ADD ../ /workdir/


