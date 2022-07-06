docker pull quay.io/biocontainers/intarna

# If image not built yet
docker build -t genetic_glitch:latest docker_unix

docker create -it \
--rm \
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