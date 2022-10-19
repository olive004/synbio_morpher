# syntax=docker/dockerfile:1

# FROM intarna_docker
# https://quay.io/repository/biocontainers/intarna
FROM quay.io/biocontainers/intarna:3.3.2--pl5321h7ff8a90_0

ARG NODE_VERSION="none"
RUN if [ "${NODE_VERSION}" != "none" ]; then su vscode -c "umask 0002 && . /usr/local/share/nvm/nvm.sh && nvm install ${NODE_VERSION} 2>&1"; fi

################################################################################
#  Install python dependencies
################################################################################

################################################################################
#  setup environment
################################################################################

# ENTRYPOINT ["/bin/bash"]