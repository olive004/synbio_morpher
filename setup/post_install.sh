#!/bin/sh

pip install jax==0.4.29
pip install jaxlib==0.4.29+cuda12.cudnn91 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
pip install -U chex
pip install git+https://github.com/Steel-Lab-Oxford/core-bioreaction-simulation.git@47391ff32aa2e0e9dcbb4541efb526ff6c43e427#egg=bioreaction
