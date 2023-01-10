docker pull docker/dockerfile:1
# docker pull quay.io/biocontainers/intarna:3.3.1--pl5321h7ff8a90_1
docker pull nvidia/cuda:11.6.2-base-ubuntu20.04

# If image not built yet
docker build -t genetic_glitch:latest docker_unix

# sudo ctr run --rm -t \
#     --runc-binary=/usr/bin/nvidia-container-runtime \
#     --env NVIDIA_VISIBLE_DEVICES=all \
#     docker.io/nvidia/cuda:11.6.2-base-ubuntu20.04 \
#     cuda-11.6.2-base-ubuntu20.04 nvidia-smi
sudo ctr create -it \
--rm \
--runc-binary=/usr/bin/nvidia-container-runtime \
--env NVIDIA_VISIBLE_DEVICES=all \
--name gcg_test \
--mount type=bind,source="$(pwd)",target=/workdir \
genetic_glitch:latest
sudo ctr container start gcg_test
sudo ctr exec -it gcg_test /bin/bash 
# docker container stop gcg


# Docusaurus
# 1. Started by installing Node.js and following installation guide https://github.com/nodejs/help/wiki/Installation
# 1.a Added to end of ~/.profile 
# 1.b Refresh . ~/.profile and test installation with npx -v