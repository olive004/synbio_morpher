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


def merge_dicts(*dict_objs):
    all_dicts = {}
    for dict_obj in dict_objs:
        if type(dict_obj) == dict:
            all_dicts = {**all_dicts, **dict_obj}
    return all_dicts


def load_json_as_dict(json_pathname):
    try:
        jdict = json.load(open(json_pathname))
        return jdict
    except FileNotFoundError:
        logging.error(f'JSON path {json_pathname} not found')
        SystemExit


def process_dict_for_json(dict_like):
    for k, v in dict_like.items():
        logging.info(v)
        logging.info(type(v))
        if type(v) == dict:
            v = process_dict_for_json(v)
        if type(v) == np.bool_:
            dict_like[k] = bool(v)
        if type(v) == np.ndarray:
            dict_like[k] = v.tolist()
    return dict_like


def get_bulkiest_dict_key(dict_like):
    k_bulkiest = list(dict_like.keys())[0]
    prev_v = dict_like[k_bulkiest]
    for k, v in dict_like.items():
        if type(v) == list:
            if len(v) > len(prev_v):
                k_bulkiest = k
    return k_bulkiest


def make_values_list(dict_like):
    return {k: [v] for k, v in dict_like.items() if not type(v) == list}


def find_sublist_max(list_like):
    list_list_sizes = [len(l) for l in list_like if type(l) == list]
    return max(list_list_sizes)


def extend_int_to_list(int_like, target_num):
    if type(int_like) == int:
        int_like = [int_like] * target_num
    elif type(int_like) == list and len(int_like) == 1:
        int_like = int_like * target_num
    return int_like


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
    logging.info(data)
    data = process_dict_for_json(data)
    logging.info(data)
    with open(out_path, 'w+') as fn:
        json.dump(data, fp=fn)
