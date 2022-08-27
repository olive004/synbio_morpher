from functools import partial
import logging
import os
import sys

from fire import Fire
from scripts.common.circuit import construct_circuit_from_cfg
from src.utils.misc.io import get_pathnames

from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main(config=None, data_writer=None):
    # set configs
    if config is None:
        config = os.path.join(
            "scripts", "generate_species_templates", "configs", "base_config_existing_circuits.json")
    config_file = load_json_as_dict(config)

    # start_experiment
    if data_writer is None:
        data_writer_kwargs = {'purpose': 'generate_species_templates'}
        data_writer = ResultWriter(**data_writer_kwargs)

    def prepare_circuit_paths_for_construction(circuit_paths):
        return [{'data_path': out_path} for out_path in circuit_paths]

    protocols = [
        Protocol(
            partial(get_pathnames,
            search_dir=config_file['source_circuits_dir'], file_key=config_file['circuits_basename']),
            req_output=True, name='get_existing_cricuits'
        ),
        Protocol(
            prepare_circuit_paths_for_construction,
            req_input=True, req_output=True, name='prepare_circuit_paths_for_construction'
        ),
        [
            Protocol(
                partial(construct_circuit_from_cfg,
                        config_filepath=config),
                req_input=True,
                req_output=True,
                name="making_circuit"
            ),
            Protocol(
                CircuitModeller(
                    result_writer=data_writer).compute_interaction_strengths,
                req_input=True,
                name="compute_interaction_strengths"
            )
        ]
    ]
    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
