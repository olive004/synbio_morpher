@REM # See README.md in the root project directory for docker setup instructions
docker pull docker/dockerfile:1
docker pull quay.io/biocontainers/intarna:3.3.1--pl5321h7ff8a90_0

@REM # If image not built yet
docker build -t genetic_glitch:latest .\docker_windows

docker create -it ^
--rm ^
--name gcg ^
--mount=type=bind,source=%cd%,target=/workdir/gcg ^
genetic_glitch:latest
docker container start gcg
docker exec -it gcg /bin/bash 
@REM # docker container stop gcg
@REM --workdir /gcg ^
@REM --volume ${PWD}:\gcg ^


@REM # Docusaurus
@REM # 1. Started by installing Node.js and following installation guide https://github.com/nodejs/help/wiki/Installation
@REM # 1.a Added to end of ~/.profile 
@REM # 1.b Refresh . ~/.profile and test installation with npx -v