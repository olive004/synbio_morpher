

from functools import partial
import logging
import os
import re
from src.utils.circuit.agnostic_circuits.circuit_manager_new import construct_circuit_from_cfg
from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.utils.results.writer import DataWriter
from src.srv.parameter_prediction.IntaRNA.bin.copomus.IntaRNA import IntaRNA
from src.utils.evolution.mutation import Evolver
from src.utils.circuit.agnostic_circuits.base_circuit import BaseSpecies
from src.utils.circuit.agnostic_circuits.circuit_manager_new import CircuitModeller


# Test non-binding of IntaRNA

def main(config=None, data_writer=None):
    if config is None:
        config = os.path.join(
            'scripts', 'intarna_sandbox', 'configs', 'intarna_no_binding.json')
    extra_configs = None

    if data_writer is None:
        data_writer = ResultWriter('intarna_sandbox')

    circuit = construct_circuit_from_cfg(
        extra_configs=extra_configs, config_filepath=config)
    simulation = CircuitModeller(result_writer=data_writer).run_interaction_simulator(circuit,
                                                                                      circuit.species.data.data)
    circuit = CircuitModeller(
        result_writer=data_writer).compute_interaction_strengths(circuit)
    logging.info(simulation.interactions)
    logging.info(circuit.species.interactions)


if __name__ == 'main':
    main()
