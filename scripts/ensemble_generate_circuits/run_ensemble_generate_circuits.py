

import os
from src.srv.io.manage.script_manager import script_preamble, ensemble_func


def main(config=None, data_writer=None):

    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "scripts", "ensemble_generate_circuits", "configs", "base_config.json"))

    ensemble_func(config, data_writer)