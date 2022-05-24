

import logging
import os
import unittest
import inspect

from src.utils.misc.io import convert_pathname_to_module, get_pathnames, get_subdirectories, import_module_from_path


SCRIPT_DIR = 'scripts'


class TestScripts(unittest.TestCase):

    def test_script_input(self):
        for script_home in get_subdirectories(SCRIPT_DIR):
            for script_name in get_pathnames('run', search_dir=script_home, allow_empty=True):
                print(script_name)
                if 'run' in script_name:
                    script = __import__(convert_pathname_to_module(script_name) + '.', 'main', fromlist=[''])
                    # script = import_module_from_path('main', script_name)
                    print(script)
                    print(convert_pathname_to_module)
                    self.assertIn('config_filepath',
                                  inspect.getfullargspec(script).args)
                    self.assertIn(
                        'writer', inspect.getfullargspec(script).args)


if __name__ == '__main__':
    unittest.main()
