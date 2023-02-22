

import os
from pathlib import Path
from typing import Union
import logging


package_name = 'gene-circuit-glitch-prediction'
if os.path.basename(os.getcwd()) == package_name:
    PACKAGE_DIR = '.'
elif os.getcwd().split(os.sep)[-2] == package_name:
    PACKAGE_DIR = '..'
else:
    PACKAGE_DIR = os.path.join(*os.getcwd().split(os.sep)[
                               :os.getcwd().split(os.sep).index(package_name)+1])

SCRIPT_DIR = os.path.join(PACKAGE_DIR, 'scripts')
DATA_DIR = os.path.join(PACKAGE_DIR, 'data')


def make_filename_safely(filename: Union[str, list]):
    if type(filename) == list:
        filename = os.path.join(**filename)
        assert os.path.isfile(filename), f'Filename {filename} is not a file - specify in config '\
            'either as list of path constituents or absolute path.'
    elif type(filename) == str:
        filename = str(Path(filename))
    return filename
