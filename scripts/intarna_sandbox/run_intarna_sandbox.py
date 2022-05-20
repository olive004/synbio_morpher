

from functools import partial
import os
from scripts.common.circuit import construct_circuit_from_cfg
from src.clients.common.setup import compose_kwargs
from src.srv.io.results.experiments import Experiment, Protocol
from src.srv.io.results.result_writer import ResultWriter
from src.srv.io.results.writer import DataWriter
from src.srv.parameter_prediction.IntaRNA.bin.copomus.IntaRNA import IntaRNA
from src.utils.evolution.mutation import Evolver
from src.utils.system_definition.agnostic_system.base_system import BaseSpecies
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


# Test non-binding of IntaRNA

def main():
    cfg_filepath = os.path.join(
        'scripts', 'intarna_sandbox', 'configs', 'intarna_no_binding.json')
    extra_configs = None

    data_writer = ResultWriter('intarna_sandbox')
    # protocols = [
    #     Protocol(
    #         partial(construct_circuit_from_cfg,
    #                 extra_configs=extra_configs, config_filepath=cfg_filepath),
    #         req_output=True,
    #         name='construct_circuit_from_cfg'
    #     ),
    #     Protocol(
    #         CircuitModeller(data_writer).compute_interaction_strengths
    #     )
    # ]

    # # Experiment(protocols=protocols)

    # circuit = construct_circuit_from_cfg(
    #     extra_configs, config_filepath=cfg_filepath)
    # modeller = CircuitModeller(data_writer)
    # modeller.compute_interaction_strengths(circuit)

    circuit = construct_circuit_from_cfg(extra_configs=extra_configs, config_filepath=cfg_filepath)
    CircuitModeller(result_writer=data_writer).compute_interaction_strengths,


if __name__ == 'main':
    main()
