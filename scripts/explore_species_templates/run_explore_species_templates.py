from functools import partial
import logging
import os
import sys

from fire import Fire
from scripts.common.circuit import construct_circuit_from_cfg

from src.srv.io.results.experiments import Experiment, Protocol
from src.srv.io.results.writer import DataWriter
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.io import get_pathnames
from src.utils.system_definition.agnostic_system.system_manager import CircuitModeller


def main():
    # set configs
    config_file = os.path.join(
        "scripts", "explore_species_templates", "configs", "explore_species_templates.json")
    exp_configs = load_json_as_dict(config_file).get("experiment")

    # start_experiment
    data_writer_kwargs = {'purpose': exp_configs.get("purpose")}
    data_writer = DataWriter(**data_writer_kwargs)
    protocols = [
        Protocol(
            # get list of all interaction paths
            partial(get_pathnames,
                    file_key="interactions",
                    root_dir="data",
                    purpose="generate_species_templates",
                    experiment_key="2022_04_14_113148",
                    subfolder="interactions"
                    ),
            name='get_pathnames'
            # do some analytics
        ),
        # read in data one at a time
        [
            Protocol(
            )
            # sort circuit into category based on number of interacting species
        ]
        # Protocol(sys.exit),
    ]
    experiment = Experiment(config_file, protocols, data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
