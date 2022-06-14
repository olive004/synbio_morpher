from functools import partial
import logging
import os

from fire import Fire

from src.srv.io.results.experiments import Experiment, Protocol
from src.srv.io.results.writer import DataWriter
from src.srv.sequence_exploration.sequence_analysis import generate_interaction_stats
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.io import get_pathnames, get_search_dir


def main(config=None, data_writer=None):
    # set configs
    if config is None:
        config = os.path.join(
            "scripts", "gather_interaction_stats", "configs", "gather_interaction_stats.json")
    config_file = load_json_as_dict(config)

    # start_experiment
    if data_writer is None:
        data_writer_kwargs = {'purpose': config_file.get(
            "experiment").get("purpose")}
        data_writer = DataWriter(**data_writer_kwargs)

    # search_dir = os.path.join(*list({
    #     'root_dir': "data",
    #     'purpose': "generate_species_templates",
    #     'experiment_key': "2022_05_10_155621",
    #     'subfolder': "interactions"
    # }.values()))

    config_file, search_dir = get_search_dir(
        "source_of_interactions", config_file=config_file)
    logging.info(search_dir)
    protocols = [
        Protocol(
            # get list of all interaction paths
            partial(get_pathnames,
                    file_key='interactions',
                    search_dir=search_dir, 
                    optional_subdir='interactions'
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
    experiment = Experiment(config_filepath=config, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
