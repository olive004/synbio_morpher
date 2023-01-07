
pip install --upgrade "jax[cuda]" -f https://storage.googleapis.com/jax-releases/jax_cuda_releases.html
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
(sleep 1; echo yes; sleep 1; echo /root/miniconda3; sleep 1; echo yes) | bash Miniconda3-latest-Linux-x86_64.sh
source /root/miniconda3/bin/activate 
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/
mkdir -p $CONDA_PREFIX/etc/conda/activate.d
echo 'export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$CONDA_PREFIX/lib/' > $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
conda deactivate