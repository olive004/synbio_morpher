#!/bin/sh

# Activate environment
CURRENT_ENV_PATH = /Users/oliviagallup/Desktop/Kode/Oxford/DPhil/env_gene_circ/bin/activate
source $CURRENT_ENV_PATH

# Configure non-pip dependencies
cd .src/utils/parameter_prediction/IntaRNA-3.2.1
./configure
make
make tests
make install prefix=$CURRENT_ENV_PATH
cd ../../..

# Clean up dependencies
cd .src/utils/parameter_prediction/IntaRNA-3.2.1
make clean
make uninstall
cd ../../..
