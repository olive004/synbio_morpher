#!/bin/sh

# Activate environment
CURRENT_ENV_PATH = /Users/oliviagallup/Desktop/Kode/Oxford/DPhil/env_gene_circ/bin/activate
source $CURRENT_ENV_PATH

# Configure non-pip dependencies
source ./src/setup/install_deps.sh

# Clean up dependencies
source ./src/setup/remove_deps.sh
