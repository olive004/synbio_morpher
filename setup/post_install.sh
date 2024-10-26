#!/bin/sh

pip install jax==0.4.29
pip install jaxlib==0.4.29+cuda12.cudnn91 -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html 
pip install -U chex