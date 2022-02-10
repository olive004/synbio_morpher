#!/bin/sh

# Activate environment
CURRENT_ENV_PATH = /Users/oliviagallup/Desktop/Kode/Oxford/DPhil/env_gene_circ/bin/activate
source $CURRENT_ENV_PATH

# Configure non-pip dependencies
## IntaRNA
cd ./src/utils/parameter_prediction/IntaRNA-3.2.1
./configure
make
make tests
make install prefix=$CURRENT_ENV_PATH --with-vrna $CURRENT_ENV_PATH --with-boost $CURRENT_ENV_PATH
cd ../../..

## ViennaRNA: Needed by IntaRNA
cd ./src/utils/parameter_prediction
tar -zxvf ViennaRNA-2.5.0.tar.gz
cd ViennaRNA-2.5.0

cd src
tar -xzf libsvm-3.25.tar.gz
tar -xjf dlib-19.22.tar.bz2
cd ..

# ViennaRNA reqs:
# gengetopt
# help2man
# flex
# flex-devel
# vim-common
# swig

./configure
make
make install prefix=$CURRENT_ENV_PATH
cd ../../..

autoreconf -i
