
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
DATA_TOP_DIR = '.'

if os.path.basename(os.getcwd()) == PACKAGE_NAME or (PACKAGE_NAME in os.listdir('.')):
    PACKAGE_DIR = os.path.join('.', PACKAGE_NAME)
elif os.getcwd().split(os.sep)[-2] == PACKAGE_NAME or (PACKAGE_NAME in os.listdir('..')):
    PACKAGE_DIR = os.path.join('..', PACKAGE_NAME)
    DATA_TOP_DIR = '..'
elif PACKAGE_NAME in os.getcwd():
    PACKAGE_DIR = os.path.join(*os.getcwd().split(os.sep)[
                               :os.getcwd().split(os.sep).index(PACKAGE_NAME)+1])
else:
    import synbio_morpher

    PACKAGE_DIR = synbio_morpher.__path__[0]

SCRIPT_DIR = 'scripts'
DATA_DIR = os.path.join(DATA_TOP_DIR, 'data')


def make_filename_safely(filename: Union[str, list]):
    if type(filename) == list:
        if DATA_DIR not in filename or (os.path.abspath(DATA_DIR) not in filename):
            filename.insert(0, DATA_DIR)
        filename = os.path.join(**filename)
        assert os.path.isfile(filename), f'Filename {filename} is not a file - specify in config '\
            'either as list of path constituents or absolute path.'
    elif type(filename) == str:
        if DATA_DIR not in filename or (os.path.abspath(DATA_DIR) not in filename):
            filename = os.path.join(DATA_DIR, filename)
        filename = str(Path(filename))
    return filename
