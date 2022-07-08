from run_parameter_grid_analysis import main as parameter_grid_analysis
from src.srv.io.manage.script_manager import Ensembler, script_preamble


def main(config=None, data_writer=None):

    config, data_writer = script_preamble(config, data_writer, alt_cfg_filepath=os.path.join(
        'scripts', 'parameter_grid_analysis', 'configs', 'heatmap_cfg.json'))


    config['']

    Ensembler(data_writer=data_writer, config=config, subscripts=)