

from copy import deepcopy
import logging
import os
from src.srv.io.results.writer import DataWriter
from src.utils.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict, write_json
from src.utils.misc.io import convert_pathname_to_module


SCRIPT_DIR = 'scripts'


def import_script_func(script_name):
    script_filepath = os.path.join(SCRIPT_DIR, script_name, f'run_{script_name}.py')
    script_module = __import__(
        convert_pathname_to_module(script_filepath), fromlist=[''])
    return getattr(script_module, 'main')


def script_preamble(config, data_writer, alt_cfg_filepath: str, use_resultwriter=True):
    writer_class = ResultWriter if use_resultwriter else DataWriter
    if config is None:
        config = alt_cfg_filepath
    config_file = load_json_as_dict(config)
    if data_writer is None:
        data_writer = writer_class(purpose=config_file['experiment']['purpose'])
    return config, data_writer


class Ensembler():

    def __init__(self, data_writer: ResultWriter, config_filepath: str, subscripts: list = None) -> None:
        self.data_writer = data_writer
        self.subscripts = subscripts

        self.config_filepath = config_filepath
        self.config = load_json_as_dict(config_filepath)
        self.ensemble_configs = self.config["base_configs_ensemble"]

    def run(self):
        for script_name in self.subscripts:
            script = import_script_func(script_name)
            logging.info(script_name)
            config = self.ensemble_configs[script_name]
            self.data_writer.update_ensemble(
                config["experiment"]["purpose"])
            # self.data_writer.update_ensemble(script_name)
            current_data_writer = deepcopy(self.data_writer)
            current_data_writer.top_write_dir = self.data_writer.ensemble_write_dir
            output = script(config, self.data_writer)
            if output:
                config, self.data_writer = output
                self.ensemble_configs[script_name] = config
        self.config["base_configs_ensemble"] = self.ensemble_configs
        write_json(self.config, self.config_filepath, overwrite=True)
