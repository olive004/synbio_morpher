
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
import json
import os
from typing import Union
import numpy as np
import pandas as pd
import jaxlib
import importlib

from synbio_morpher.srv.io.manage.sys_interface import PACKAGE_NAME
from synbio_morpher.utils.misc.type_handling import inverse_dict


FORMAT_EXTS = {
    "fasta": "fasta",
    "npy": "numpy",
    "csv": "csv",
    "json": "json"
}


def verify_file_type(filepath: str, file_type: str):
    assert file_type in filepath or inverse_dict(FORMAT_EXTS).get(file_type) in filepath, \
        f'File {filepath} is not of type {file_type}'


def determine_file_format(filepath: str) -> str:
    ext = os.path.basename(filepath).split('.')[-1]
    try:
        return FORMAT_EXTS[ext]
    except KeyError:
        if os.path.isfile(filepath):
            raise KeyError(f'Could not determine the file format of file {filepath}: "{ext}" '
                           f'not in acceptable extensions {FORMAT_EXTS.keys()}')
        else:
            raise ValueError(
                f'Attempted to determine file format of an object that is not a file: {filepath}')


def load_json_as_dict(json_pathname: str, process=True) -> dict:
    if not json_pathname:
        return {}
    elif type(json_pathname) == dict:
        jdict = json_pathname
    elif type(json_pathname) == str:
        if PACKAGE_NAME in json_pathname:
            stripped = json_pathname.strip(f'..{os.sep}').strip(f'.{os.sep}')
            as_module = '.'.join(stripped.split(os.sep)[:-1])
            full_json_pathname = os.path.join(
                importlib.import_module(as_module).__path__[0],
                os.path.basename(json_pathname))
        else: 
            full_json_pathname = json_pathname
        if os.stat(full_json_pathname).st_size == 0:
            jdict = {}
        else:
            file = open(full_json_pathname)
            jdict = json.load(file)
            file.close()
    else:
        raise TypeError(
            f'Unknown json loading input {json_pathname} of type {type(json_pathname)}.')
    if process:
        return process_json(jdict)
    return jdict


def load_json_as_df(json_pathname: str) -> pd.DataFrame:
    return pd.read_json(json_pathname)


def load_multiple_as_list(inputs_list: list, load_func, **kwargs) -> list:
    collection_list = []
    assert type(
        inputs_list) == list, f'Input type of {inputs_list} should be list and not {type(inputs_list)}'
    for inp in inputs_list:
        collection_list.append(load_func(inp, **kwargs))
    return collection_list


def load_csv_mult(file_paths) -> list:
    return load_multiple_as_list(file_paths, load_csv)


def load_json_mult(file_paths: list, as_type=dict) -> list:
    if as_type == dict:
        return load_multiple_as_list(file_paths, load_json_as_dict)
    elif as_type == pd.DataFrame:
        return load_multiple_as_list(file_paths, load_json_as_df)


def load_csv(file_path, **kwargs) -> pd.DataFrame:
    loaded = pd.read_csv(file_path, **kwargs)
    return loaded


def make_iterable_like(dict_like):
    if type(dict_like) == dict:
        iterable_like = dict_like.items()
    elif type(dict_like) == list:
        iterable_like = enumerate(dict_like)
    else:
        raise ValueError(
            f'Input {dict_like} should be type dict (or list) but was type {type(dict_like)}')
    return iterable_like


def process_dict_for_json(dict_like) -> Union[list, dict]:
    iterable_like = make_iterable_like(dict_like)
    for k, v in iterable_like:
        if type(v) == dict or type(v) == list:
            dict_like[k] = process_dict_for_json(v)
        elif type(v) == np.bool_:
            dict_like[k] = bool(v)
        elif type(v) == np.ndarray or type(v) in jaxlib.xla_extension.__dict__.values():
            dict_like[k] = v.tolist()
        elif type(v) == np.float32 or type(v) == np.int64 or type(v) == np.float16:
            dict_like[k] = str(v)
    return dict_like


def process_json(json_dict):
    for k, v in json_dict.items():
        if v == "None":
            json_dict[k] = None
    return json_dict


def write_csv(data: pd.DataFrame, out_path: str, overwrite=False):
    import jaxlib
    if type(data) == dict:
        data = {k: [v] for k, v in data.items()}
        data = pd.DataFrame.from_dict(data, dtype=object)
    if type(data) == pd.DataFrame:
        if overwrite or not os.path.exists(out_path):
            data.to_csv(out_path, index=None)
        else:
            data.to_csv(out_path, mode='a', header=None, index=None) 
    elif type(data) == np.ndarray or type(data) == jaxlib.xla_extension.DeviceArray or type(data) == jaxlib.xla_extension.ArrayImpl:
        pd.DataFrame(data).to_csv(out_path, mode='a', header=None, index=None)
    else:
        raise TypeError(
            f'Unsupported: cannot output data of type {type(data)} to csv.')


def write_json(data: Union[dict, pd.DataFrame], out_path: str, overwrite=False):
    if type(data) == pd.DataFrame:
        data.reset_index(drop=True, inplace=True)
        data.to_json(out_path)
    else:
        data = process_dict_for_json(data)
        with open(out_path, 'w+') as fn:
            json.dump(data, fp=fn, indent=4)


def write_np(data: np.array, out_path: str, overwrite=False):
    if not overwrite and os.path.exists(out_path):
        return
    np.save(file=out_path, arr=data)
