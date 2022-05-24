

import logging
import os
import unittest
import inspect

from src.utils.misc.io import convert_pathname_to_module, get_pathnames, get_subdirectories


SCRIPT_DIR = 'scripts'


class TestScripts(unittest.TestCase):

    def test_script_input(self):
        for script_home in get_subdirectories(SCRIPT_DIR):
            for script_name in get_pathnames('run', search_dir=script_home, allow_empty=True):
                script_module = __import__(convert_pathname_to_module(script_name), fromlist=[''])
                script = getattr(script_module, 'main')
                if script.__name__ == 'wrapper':
                    logging.warning(f'Could not test `main` function in {script_name} because it is wrapped.')
                else:
                    self.assertIn('config_filepath',
                                inspect.getfullargspec(script).args)
                    self.assertIn(
                        'data_writer', inspect.getfullargspec(script).args)


if __name__ == '__main__':
    unittest.main()
