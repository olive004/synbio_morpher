



import logging
import os
import unittest
import inspect

import numpy as np
from src.clients.common.setup import compose_kwargs
from src.utils.circuit.agnostic_circuits.base_circuit import BaseCircuit
from src.utils.circuit.common.config_setup import parse_cfg_args
from src.utils.data.common import Data


class TestCircuits(unittest.TestCase):

    def test_base_circuit(self):
        config_args = {
            "data": Data(loaded_data={
                "RNA0": "AUCUGGUAACUCC"
            }),
            "identities": {
                "input": "RNA0"
            },
            "molecular_params": {
                "starting_copynumbers": 0,
                "creation_rate": 0,
                "degradation_rate": 0
            },
            "interaction_simulator": {
                "name": "IntaRNA"
            },
            "system_type": "RNA"
        }

        config_args = parse_cfg_args(compose_kwargs(config_file=config_args))
        base_circuit = BaseCircuit(config_args=config_args)





if __name__ == '__main__':
    unittest.main()