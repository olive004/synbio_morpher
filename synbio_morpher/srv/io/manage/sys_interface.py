
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


import os
from pathlib import Path
from typing import Union
import logging


# Relative imports
PACKAGE_NAME = 'synbio_morpher'
# if PACKAGE_NAME not in os.getcwd():
#     if 'workdir' in os.getcwd():
#         PACKAGE_NAME = 'workdir'

if os.path.basename(os.getcwd()) == PACKAGE_NAME:
    PACKAGE_DIR = '.'
elif os.getcwd().split(os.sep)[-2] == PACKAGE_NAME:
    PACKAGE_DIR = '..'
elif PACKAGE_NAME in os.getcwd():
    PACKAGE_DIR = os.path.join(*os.getcwd().split(os.sep)[
                               :os.getcwd().split(os.sep).index(PACKAGE_NAME)+1])
else:
    PACKAGE_DIR = '.'
    logging.warning(
        f'Package name {PACKAGE_NAME} not found in current working directory {os.getcwd()} - will create dirs in cwd when necessary.')


SCRIPT_DIR = 'scripts'
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
