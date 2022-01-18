# docker pull biopython/biopython


docker create -it \
--rm \
--name gcg \
--mount type=bind,source="$(pwd)",target=/workdir/gcg \
gene_circuit_glitching:latest
docker container start gcg
# docker exec -it gcg /bin/bash
# docker container stop gcg