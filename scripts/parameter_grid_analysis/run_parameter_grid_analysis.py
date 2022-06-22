


import os
from src.srv.io.results.result_writer import ResultWriter
from src.utils.data.data_format_tools.common import load_json_as_dict


def main(config=None, data_writer=None):

    if config is None:
        config = os.path.join(
            'scripts', 'parameter_based_simulation', 'configs', 'medium_parameter_space.json')
    config_file = load_json_as_dict(config)
    if data_writer is None:
        data_writer = ResultWriter(purpose=config_file.get(
            'experiment').get('purpose', 'parameter_based_simulation'))

    # Load in parameter grids


    # stitch them together 
        # Find the starting and ending indices

        # 