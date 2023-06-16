
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
from fire import Fire

from src.utils.misc.decorators import time_it
from src.utils.circuit.agnostic_circuits.base_circuit import BaseCircuit


@time_it
def main(config_args=None):
    circuit = BaseCircuit(config_args)
    circuit.visualise()

if __name__ == "__main__":
    Fire(main)
