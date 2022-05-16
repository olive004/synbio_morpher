

import os
from scripts.common.circuit import construct_circuit_from_cfg
from src.srv.io.results.experiments import Experiment, Protocol
from src.srv.io.results.writer import DataWriter
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


# Test non-binding of IntaRNA

def main():
    cfg_filepath = os.path.join(
        'scripts', 'sandbox_simulators', 'configs', 'intarna_no_binding.json')

    data_writer = DataWriter('intarna_sandbox')
    protocols = [
        Protocol(
            construct_circuit_from_cfg(cfg_filepath),
            req_output=True,
            name='construct_circuit_from_cfg'
        ),
        Protocol(
            CircuitModeller(data_writer).compute_interaction_strengths
        )
    ]

    Experiment(protocols=protocols)


    circuit = construct_circuit_from_cfg(cfg_filepath)
    modeller = CircuitModeller(data_writer)
    modeller.compute_interaction_strengths(circuit)


if __name__ == 'main':
    main()
