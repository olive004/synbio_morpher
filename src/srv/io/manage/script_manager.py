

from copy import deepcopy
from datetime import datetime
import logging
import os
from src.utils.results.writer import DataWriter
from src.utils.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.io import convert_pathname_to_module


SCRIPT_DIR = 'scripts'


def import_script_func(script_name):
    script_filepath = os.path.join(
        SCRIPT_DIR, script_name, f'run_{script_name}.py')
    if not os.path.isfile(script_filepath):
        return None
    script_module = __import__(
        convert_pathname_to_module(script_filepath), fromlist=[''])
    return getattr(script_module, 'main')


def script_preamble(config, data_writer, alt_cfg_filepath: str = None, use_resultwriter=True) -> tuple:
    Writer = ResultWriter if use_resultwriter else DataWriter
    if config is None:
        config = alt_cfg_filepath
    config_file = load_json_as_dict(config)
    if data_writer is None:
        data_writer = Writer(purpose=config_file['experiment']['purpose'])
    return config, data_writer


class Ensembler():

    def __init__(self, data_writer: ResultWriter, config: str) -> None:
        self.data_writer = data_writer

        self.config = load_json_as_dict(config)
        self.ensemble_configs = self.config["base_configs_ensemble"]
        self.subscripts = [script for script in self.ensemble_configs.keys()]
        self.start_time = datetime.now()

    def run(self):
        for script_name in self.subscripts:
            script = import_script_func(script_name)
            if not script:
                continue
            logging.warning(f'\tRunning script {script_name}\n')
            config = self.ensemble_configs[script_name]
            if config["experiment"]["purpose"] != script_name:
                logging.warning(
                    f'The current script being run {script_name} does not match its '
                    f'config purpose {config["experiment"]["purpose"]}')
            self.data_writer.update_ensemble(
                config["experiment"]["purpose"])
            # self.data_writer.update_ensemble(script_name)
            # current_data_writer = deepcopy(self.data_writer)
            # current_data_writer.top_write_dir = self.data_writer.ensemble_write_dir
            output = script(config, self.data_writer)
            if output:
                config, self.data_writer = output
                self.ensemble_configs[script_name] = config
        self.config["base_configs_ensemble"] = self.ensemble_configs

        self.data_writer.reset_ensemble()
        self.config['total_time'] = str(datetime.now() - self.start_time)
        self.write()

    def write(self):
        self.data_writer.output(
            out_type='json', out_name='ensemble_config', data=self.config, write_master=False,
            write_to_top_dir=True)
