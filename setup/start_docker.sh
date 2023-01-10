docker pull docker/dockerfile:1
# docker pull quay.io/biocontainers/intarna:3.3.1--pl5321h7ff8a90_1
docker pull nvidia/cuda:11.6.2-base-ubuntu20.04

# If image not built yet
docker build -t genetic_glitch:latest docker_unix


docker create -it \
--rm \
--gpus all \
--name gcg_test \
--mount type=bind,source="$(pwd)",target=/workdir \
genetic_glitch:latest
docker container start gcg_test
docker exec -it gcg_test /bin/bash 
# docker container stop gcg


# Docusaurus
# 1. Started by installing Node.js and following installation guide https://github.com/nodejs/help/wiki/Installation
# 1.a Added to end of ~/.profile 
# 1.b Refresh . ~/.profile and test installation with npx -v