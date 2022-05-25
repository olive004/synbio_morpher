

import logging
import os
import unittest
import inspect
from src.utils.data.data_format_tools.common import load_json_as_dict

from src.utils.misc.io import convert_pathname_to_module, get_pathnames, get_subdirectories


SCRIPT_DIR = 'scripts'


def get_all_config_files(config_dir):
    flat_configs = []
    dirs = get_subdirectories(config_dir)
    for config in get_pathnames(config_dir):
        if config in dirs:
            flat_configs = flat_configs + get_all_config_files(config)
        else:
            flat_configs.append(config)
    return flat_configs


class TestScripts(unittest.TestCase):

    def test_script_input(self):
        for script_home in get_subdirectories(SCRIPT_DIR):
            for script_name in get_pathnames(file_key='run', search_dir=script_home, allow_empty=True):
                script_module = __import__(
                    convert_pathname_to_module(script_name), fromlist=[''])
                script = getattr(script_module, 'main')
                if script.__name__ == 'wrapper':
                    logging.warning(
                        f'Could not test `main` function in {script_name} because it is wrapped.')
                else:
                    self.assertIn('config_filepath',
                                  inspect.getfullargspec(script).args)
                    self.assertIn(
                        'data_writer', inspect.getfullargspec(script).args)

    def test_script_baseconfig(self):

        base_config = {}
        for script_home in get_subdirectories(SCRIPT_DIR):
            self.assertIn('configs', get_subdirectories(script_home))
            for config_path in get_all_config_files(os.path.join(script_home, 'configs')):
                if 'base_config' in config_path:
                    base_config = load_json_as_dict(config_path)
                config = load_json_as_dict(config_path)
                
                self.assertTrue(all(key in config.keys() for key in base_config.keys()))
                # all(item in superset.items() for item in subset.items())


if __name__ == '__main__':
    unittest.main()
