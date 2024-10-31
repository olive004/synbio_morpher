
# A pre-req for using gpus here is the NVIDIA Docker Container Toolkit

sudo docker pull docker/dockerfile:1
sudo docker pull quay.io/biocontainers/intarna:3.3.2--pl5321h7ff8a90_0
# sudo docker run --rm -it --entrypoint bash quay.io/biocontainers/intarna:3.3.2--pl5321h7ff8a90_0
# sudo docker pull nvidia/cuda:12.1.0-devel-ubuntu22.04

# If image not built yet
if [ "$(id -un)" != "wadh6511" ]; then
    sudo docker pull nvidia/cuda:12.0.0-cudnn8-devel-ubuntu20.04
    source_directory="docker_unix_hslab"
else
    sudo docker pull nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04
    source_directory="docker_unix"
fi

sudo docker build -t genetic_glitch:latest $source_directory

# sudo ctr run --rm -t \
#     --runc-binary=/usr/bin/nvidia-container-runtime \
#     --env NVIDIA_VISIBLE_DEVICES=all \
#     sudo docker.io/nvidia/cuda:11.6.2-base-ubuntu20.04 \
#     cuda-11.6.2-base-ubuntu20.04 nvidia-smi
cp ./requirements.txt ./$source_directory

sudo docker create -it \
--rm \
--gpus all \
--name gcg \
--mount type=bind,source="$(pwd)",target=/workdir \
genetic_glitch:latest
sudo docker container start gcg
sudo docker exec -it gcg bash setup/post_install.sh
sudo docker exec -it gcg /bin/bash
# sudo docker container stop gcg

if [ "$(id -un)" != "wadh6511" ]; then
    bash $source_directory/post_install2.sh
fi

# Docusaurus
# 1. Started by installing Node.js and following installation guide https://github.com/nodejs/help/wiki/Installation
# 1.a Added to end of ~/.profile 
# 1.b Refresh . ~/.profile and test installation with npx -v