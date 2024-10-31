#!/bin/sh

# Install python3.10
apt update
apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install python3.10
apt install python3.10-venv
apt install python3.10-dev
ls -la /usr/bin/python3
rm /usr/bin/python3
ln -s python3.10 /usr/bin/python3
python3 --version
apt install curl
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
# python3.10 -m pip install ipython