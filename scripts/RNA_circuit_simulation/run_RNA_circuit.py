

from functools import partial
import logging
import os
from fire import Fire
from scripts.common.circuit import construct_circuit_from_cfg
from src.srv.io.manage.script_manager import script_preamble
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.decorators import time_it
from src.utils.results.experiments import Experiment, Protocol
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


@time_it
def main(config=None, data_writer=None):

    # from src.utils.data.fake_data_generation.seq_generator import RNAGenerator
    # RNAGenerator(purpose='example_data').generate_circuit(
    #     count=3, slength=25, protocol="template_mix")

    config, data_writer = script_preamble(config=config, data_writer=data_writer, alt_cfg_filepath=os.path.join(
        "scripts", "RNA_circuit_simulation", "configs", "toy_RNA.json"))
    config_file = load_json_as_dict(config)
    logging.info(config)
    logging.info(config_file)

    def readout(inp):
        logging.info('\n\n\n\nReadout')
        logging.info(inp)
        return inp

    protocols = [
        Protocol(partial(
            construct_circuit_from_cfg,
            extra_configs=None, config_filepath=config),
            req_output=True, name='construct_circuit_from_cfg'),
        Protocol(readout, req_output=True, req_input=True),
        Protocol(partial(
            CircuitModeller(result_writer=data_writer,
                            config=config_file).apply_to_circuit,
            methods={
                'init_circuit': {},
                'simulate_signal': {
                    'use_solver': load_json_as_dict(
                        config_file.get('signal')).get('use_solver', 'naive')},
                'write_results': {}}),
            req_input=True, name='model circuit and write results')
    ]
    experiment = Experiment(config=config, config_file=config_file,
                            protocols=protocols, data_writer=data_writer)
    experiment.run_experiment()

    # circuit = construct_circuit_from_cfg(None, config_filepath=config)
    # modeller = CircuitModeller(result_writer=data_writer)
    # circuit = modeller.init_circuit(circuit)
    # circuit = modeller.simulate_signal(
    #     circuit, use_solver=config_file.get('signal').get('use_solver', 'naive'))

    # modeller.write_results(circuit)


if __name__ == "__main__":
    Fire(main)
