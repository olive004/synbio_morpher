# docker pull quay.io/biocontainers/intarna

# If image not built yet
docker build -t genetic_glitch:latest docker

docker create -it \
--rm \
--name gcg \
--mount type=bind,source="$(pwd)",target=/workdir/gcg \
genetic_glitch:latest
docker container start gcg
docker exec -it gcg /bin/bash
# docker container stop gcg