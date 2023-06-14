
# A pre-req for using gpus here is the NVIDIA Docker Container Toolkit

sudo docker pull docker/dockerfile:1
sudo docker pull quay.io/biocontainers/intarna:3.3.2--pl5321h7ff8a90_0
# sudo docker pull nvidia/cuda:12.1.0-devel-ubuntu22.04
sudo docker pull nvidia/cuda:11.8.0-cudnn8-devel-ubuntu22.04

# If image not built yet
sudo docker build -t genetic_glitch:latest docker_unix

# sudo ctr run --rm -t \
#     --runc-binary=/usr/bin/nvidia-container-runtime \
#     --env NVIDIA_VISIBLE_DEVICES=all \
#     sudo docker.io/nvidia/cuda:11.6.2-base-ubuntu20.04 \
#     cuda-11.6.2-base-ubuntu20.04 nvidia-smi
cp ./requirements.txt ./docker_unix

sudo docker create -it \
--rm \
--gpus all \
--name gcg \
--mount type=bind,source="$(pwd)",target=/workdir \
genetic_glitch:latest
sudo docker container start gcg
sudo docker exec -it gcg /bin/bash 
# sudo docker container stop gcg


# Docusaurus
# 1. Started by installing Node.js and following installation guide https://github.com/nodejs/help/wiki/Installation
# 1.a Added to end of ~/.profile 
# 1.b Refresh . ~/.profile and test installation with npx -v