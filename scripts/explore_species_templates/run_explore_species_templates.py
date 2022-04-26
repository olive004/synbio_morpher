from functools import partial
import os

from fire import Fire

from src.srv.io.results.experiments import Experiment, Protocol
from src.srv.io.results.writer import DataWriter
from src.srv.sequence_exploration.sequence_analysis import generate_interaction_stats
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.io import get_pathnames


def main():
    # set configs
    config_filepath = os.path.join(
        "scripts", "explore_species_templates", "configs", "explore_species_templates.json")
    exp_configs = load_json_as_dict(config_filepath).get("experiment")

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
                    experiment_key="2022_04_26_180056",
                    subfolder="interactions"
                    ),
            req_output=True,
            name='get_pathnames'
        ),
        # read in data one at a time
        [
            # do some analytics
            Protocol(
                partial(generate_interaction_stats, writer=data_writer),
                req_output=True,
                req_input=True,
                name='analyse_interactions'
            )
        ]
    ]
    experiment = Experiment(config_filepath, protocols, data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
