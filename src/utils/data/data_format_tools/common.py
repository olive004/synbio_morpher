import json
import logging
import os
import numpy as np
import pandas as pd


FORMAT_EXTS = {
    ".fasta": "fasta"
}


def verify_file_type(filepath: str, file_type: str):
    NotImplemented
    pass


def determine_data_format(filepath):
    return os.path.basename(filepath).split('.')[-1]
    # for extension, ftype in FORMAT_EXTS.items():
    #     if extension in filepath:
    #         return ftype
    # return None


def load_json_as_dict(json_pathname):
    if not json_pathname:
        return {}
    try:
        jdict = json.load(open(json_pathname))
        return jdict
    except FileNotFoundError:
        logging.error(f'JSON path {json_pathname} not found')
        import sys
        sys.exit()


def process_dict_for_json(dict_like):
    for k, v in dict_like.items():
        if type(v) == dict:
            v = process_dict_for_json(v)
        if type(v) == np.bool_:
            dict_like[k] = bool(v)
        if type(v) == np.ndarray:
            dict_like[k] = v.tolist()
    return dict_like


def process_json(json_dict):
    for k, v in json_dict.items():
        if v == "None":
            json_dict[k] = None


def write_csv(data: pd.DataFrame, out_path: str, overwrite=False):
    if type(data) == dict:
        data = {k: [v] for k, v in data.items()}
        data = pd.DataFrame.from_dict(data)
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
        json.dump(data, fp=fn, indent=4)
