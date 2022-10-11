from functools import partial
import os

from fire import Fire
from src.srv.io.manage.script_manager import script_preamble

from src.utils.results.experiments import Experiment, Protocol
from src.utils.results.result_writer import ResultWriter
from src.srv.sequence_exploration.sequence_analysis import tabulate_mutation_info
from src.utils.data.data_format_tools.common import load_json_as_dict, load_multiple_as_list


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
            "scripts", "summarise_simulation", "configs", "analyse_mutated_templates_20_highmag.json"))
    config_file = load_json_as_dict(config)

    # Start_experiment
    if data_writer is None:
        data_writer = ResultWriter(purpose=config_file.get('experiment', {}).get('purpose'))

    source_dirs = config_file.get('source_dirs')
    protocols = [
        Protocol(
            partial(load_multiple_as_list, inputs_list=source_dirs, load_func=tabulate_mutation_info, 
            data_writer=data_writer),
            req_output=True,
            name='tabulate_mutation_info'
        ),
        Protocol(
            partial(
                get_pathnames_from_mult_dirs,
                search_dirs=source_dirs,
                file_key='tabulated_mutation_info.json',
                first_only=True),
            req_output=True,
            name='get_pathnames_from_mult_dirs'
        ),
        Protocol(
            partial(
                load_json_mult,
                as_type=pd.DataFrame
            ),
            req_input=True,
            req_output=True,
            name='load_json'
        ),
        Protocol(
            partial(
                pd.concat,
                axis=0,
                ignore_index=True),
            req_input=True,
            req_output=True,
            name='concatenate_dfs'
        ),
    ]
    experiment = Experiment(config=config, config_file=config_file, protocols=protocols,
                            data_writer=data_writer)
    experiment.run_experiment()


if __name__ == "__main__":
    Fire(main)
