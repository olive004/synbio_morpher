import json
import logging
import pandas as pd


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


def get_bulkiest_dict_key(dict_like):
    k_bulkiest = list(dict_like.keys())[0]
    prev_v = dict_like[k_bulkiest]
    for k, v in dict_like.items():
        if type(v) == list:
            if len(v) > len(prev_v):
                k_bulkiest = k
    return k_bulkiest


def find_list_max(list_like):
    list_list_sizes = [len(l) for l in list_like if type(l) == list]
    return max(list_list_sizes)


def write_csv(data, path_name):
    if type(data) == pd.DataFrame:
        data.to_csv(path_name)
    else:
        raise TypeError(f'Unsupported: cannot output data of type {type(data)} to csv.')
    
