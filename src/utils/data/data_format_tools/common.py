import json
import logging
import os
import numpy as np
import pandas as pd

from src.utils.misc.string_handling import add_outtype, make_time_str


FORMAT_EXTS = {
    ".fasta": "fasta"
}


def verify_file_type(filepath: str, file_type: str):
    NotImplemented
    pass


def determine_data_format(filepath):
    for extension, ftype in FORMAT_EXTS.items():
        if extension in filepath:
            return ftype
    return None


def load_json_as_dict(json_pathname):
    try:
        jdict = json.load(open(json_pathname))
        return jdict
    except FileNotFoundError:
        logging.error(f'JSON path {json_pathname} not found')
        SystemExit


def process_dict_for_json(dict_like):
    for k, v in dict_like.items():
        if type(v) == dict:
            v = process_dict_for_json(v)
        if type(v) == np.bool_:
            dict_like[k] = bool(v)
        if type(v) == np.ndarray:
            dict_like[k] = v.tolist()
    return dict_like


def write_csv(data: pd.DataFrame, out_path: str, overwrite=False, new_vis=False, out_type='csv'):
    if new_vis:
        out_path = f'{out_path}_{make_time_str()}'
    out_path = add_outtype(out_path, out_type)
    if type(data) == pd.DataFrame:
        if overwrite or not os.path.exists(out_path):
            data.to_csv(out_path, index=False)
        else:
            data.to_csv(out_path, mode='a', header=False, index=False)
    else:
        raise TypeError(
            f'Unsupported: cannot output data of type {type(data)} to csv.')


def write_json(data: dict, out_path: str, overwrite=False, out_type='json'):
    data = process_dict_for_json(data)
    with open(out_path, 'w+') as fn:
        json.dump(data, fp=fn)
