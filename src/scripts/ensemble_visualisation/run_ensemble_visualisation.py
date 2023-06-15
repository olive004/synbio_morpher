import os
from src.srv.io.manage.script_manager import script_preamble, ensemble_func


def main(config=None, data_writer=None):
    # Set configs
    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        "src", "scripts", "ensemble_visualisation", "configs", "base_config.json"))
        
    ensemble_func(config, data_writer)
