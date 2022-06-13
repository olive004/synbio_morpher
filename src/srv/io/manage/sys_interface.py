

import os
from pathlib import Path
from typing import Union


def make_filename_safely(filename: Union[str, list]):
    if type(filename) == list:
        filename = os.path.join(**filename)
        assert os.path.isfile(filename), f'Filename {filename} is not a file - specify in config '\
            'either as list of path constituents or absolute path.'
    elif type(filename) == str:
        filename = str(Path(filename))
    return filename