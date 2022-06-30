

from functools import partial
import logging
import os
import re
from scripts.common.circuit import construct_circuit_from_cfg
from src.clients.common.setup import compose_kwargs
from src.src.utils.results.experiments import Experiment, Protocol
from src.src.utils.results.result_writer import ResultWriter
from src.src.utils.results.writer import DataWriter
from src.srv.parameter_prediction.IntaRNA.bin.copomus.IntaRNA import IntaRNA
from src.srv.parameter_prediction.simulator import simulate_intaRNA_data
from src.utils.evolution.mutation import Evolver
from src.utils.system_definition.agnostic_system.base_system import BaseSpecies
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


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
    logging.info(simulation.data)
    logging.info(circuit.species.interactions)


if __name__ == 'main':
    main()
