#!/bin/sh

# Clean up dependencies
## IntaRNA
cd ./src/utils/parameter_prediction/IntaRNA-3.2.1
make clean
make uninstall
cd ../../..

## ViennaRNA
cd ./src/utils/parameter_prediction/ViennaRNA-2.5.0
make clean
make uninstall
cd ../../..
