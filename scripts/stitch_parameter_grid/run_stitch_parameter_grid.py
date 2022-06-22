


import os
from src.srv.io.loaders.data_loader import DataLoader
from src.srv.io.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict
from src.utils.misc.io import get_search_dir


def main(config=None, data_writer=None):

    if config is None:
        config = os.path.join(
            'scripts', 'stitch_parameter_grid', 'configs', 'base_config.json')
    config_file = load_json_as_dict(config)
    if data_writer is None:
        data_writer = ResultWriter(purpose=config_file.get(
            'experiment').get('purpose', 'stitch_parameter_grid'))

    # Load in parameter grids

    source_dir = get_search_dir('source_parameter_dir', config_file=config_file)
    parameter_grids = DataLoader().load_data(source_dir)

    # stitch them together 
        # Find the starting and ending indices

    # Write full matrices