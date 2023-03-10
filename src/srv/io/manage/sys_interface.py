

import os
from pathlib import Path
from typing import Union
import logging


# Relative imports
package_name = 'gene-circuit-glitch-prediction'
if package_name not in os.getcwd():
    if 'workdir' in os.getcwd():
        package_name = 'workdir'

if os.path.basename(os.getcwd()) == package_name:
    PACKAGE_DIR = '.'
elif os.getcwd().split(os.sep)[-2] == package_name:
    PACKAGE_DIR = '..'
elif package_name in os.getcwd():
    PACKAGE_DIR = os.path.join(*os.getcwd().split(os.sep)[
                               :os.getcwd().split(os.sep).index(package_name)+1])
else:
    logging.warning(
        f'Package name {package_name} not found in current working directory {os.getcwd()}')


SCRIPT_DIR = os.path.join(PACKAGE_DIR, 'scripts')
DATA_DIR = os.path.join(PACKAGE_DIR, 'data')


def make_filename_safely(filename: Union[str, list]):
    if type(filename) == list:
        if PACKAGE_DIR not in filename:
            filename.insert(0, PACKAGE_DIR)
        filename = os.path.join(**filename)
        assert os.path.isfile(filename), f'Filename {filename} is not a file - specify in config '\
            'either as list of path constituents or absolute path.'
    elif type(filename) == str:
        if PACKAGE_DIR not in filename:
            filename = os.path.join(PACKAGE_DIR, filename)
        filename = str(Path(filename))
    return filename
