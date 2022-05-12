from functools import partial
import logging
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

    # start_experiment
    data_writer_kwargs = {'purpose': load_json_as_dict(config_filepath).get("experiment").get("purpose")}
    data_writer = DataWriter(**data_writer_kwargs)
    search_dir = os.path.join(*list({
        'root_dir': "data",
        'purpose': "generate_species_templates",
        'experiment_key': "2022_05_10_155621",
        'subfolder': "interactions"
    }.values()))
    protocols = [
        Protocol(
            # get list of all interaction paths
            partial(get_pathnames,
                    file_key="interactions",
                    search_dir=search_dir
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
    experiment = Experiment(config_filepath, protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
