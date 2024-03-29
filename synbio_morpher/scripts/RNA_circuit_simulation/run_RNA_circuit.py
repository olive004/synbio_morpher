
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


from functools import partial
import logging
import os
from fire import Fire
from synbio_morpher.srv.io.manage.script_manager import script_preamble
from synbio_morpher.utils.common.setup import construct_circuit_from_cfg, prepare_config
from synbio_morpher.utils.data.data_format_tools.common import load_json_as_dict
from synbio_morpher.utils.misc.decorators import time_it
from synbio_morpher.utils.results.experiments import Experiment, Protocol
from synbio_morpher.utils.circuit.agnostic_circuits.circuit_manager import CircuitModeller


@time_it
def main(config=None, data_writer=None):

    config, data_writer = script_preamble(config=config, data_writer=data_writer, alt_cfg_filepath=os.path.join(
        "synbio_morpher", "scripts", "RNA_circuit_simulation", "configs", "toy_RNA.json"))
    config_file = load_json_as_dict(config)
    config_file = prepare_config(config_file)
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


if __name__ == "__main__":
    Fire(main)
