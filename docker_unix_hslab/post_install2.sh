#!/bin/sh

# Install python3.10
apt update
apt install software-properties-common -y
add-apt-repository ppa:deadsnakes/ppa
apt update
apt install python3.11
apt install python3.11-venv
apt install python3.11-dev
ls -la /usr/bin/python3
rm /usr/bin/python3
ln -s python3.11 /usr/bin/python3
apt install curl
curl -sS https://bootstrap.pypa.io/get-pip.py | python3.11
python3 --version
# python3.10 -m pip install ipython

# pip install git+https://github.com/Steel-Lab-Oxford/core-bioreaction-simulation.git@599990dcac56e7678f45269d2fc9df736d25f356#egg=bioreaction
pip install -e src/bioreaction
