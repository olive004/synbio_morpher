
# A pre-req for using gpus here is the NVIDIA Docker Container Toolkit

docker pull docker/dockerfile:1
docker pull quay.io/biocontainers/intarna:3.3.2--pl5321h7ff8a90_0
docker pull nvidia/cuda:11.4.0-base-ubuntu20.04

# If image not built yet
docker build -t genetic_glitch:latest docker_unix

# sudo ctr run --rm -t \
#     --runc-binary=/usr/bin/nvidia-container-runtime \
#     --env NVIDIA_VISIBLE_DEVICES=all \
#     docker.io/nvidia/cuda:11.6.2-base-ubuntu20.04 \
#     cuda-11.6.2-base-ubuntu20.04 nvidia-smi
docker create -it \
--rm \
--gpus all \
--name gcg \
--mount type=bind,source="$(pwd)",target=/workdir \
genetic_glitch:latest
docker container start gcg
docker exec -it gcg /bin/bash 
# docker container stop gcg


# Docusaurus
# 1. Started by installing Node.js and following installation guide https://github.com/nodejs/help/wiki/Installation
# 1.a Added to end of ~/.profile 
# 1.b Refresh . ~/.profile and test installation with npx -v