

from src.srv.io.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict, write_json


class Ensembler():

    def __init__(self, data_writer: ResultWriter, config_filepath: str, subscripts: list = None) -> None:
        self.data_writer = data_writer
        self.subscripts = subscripts

        self.config_filepath = config_filepath
        self.config_file = load_json_as_dict(config_filepath)
        self.ensemble_configs = self.config_file.get("base_configs_ensemble", {})

    def run(self):
        for script in self.subscripts:
            config = self.ensemble_configs[script.__name__]
            self.data_writer.update_ensemble(
                config.get("experiment").get("purpose"))
            # self.data_writer.update_ensemble(script.__name__)
            output = script(config, self.data_writer)
            if output:
                config, self.data_writer = output
                self.ensemble_configs[script.__name__] = config
        self.config_file["base_configs_ensemble"] = self.ensemble_configs
        write_json(self.config_file, self.config_filepath, overwrite=True)
