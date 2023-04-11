

import logging
from typing import List, Union
from copy import deepcopy


def assert_uniform_type(list_like, target_type):
    assert all(type(list_like) == target_type)


def cast_all_values_as_list(dict_like):
    new_dict = {}
    for k, v in dict_like.items():
        if not type(v) == list:
            new_dict[k] = [v]
        else:
            new_dict[k] = v
    return new_dict
    # return {k: [v] for k, v in dict_like.items() if not type(v) == list}
    # return {k: [v] for k, v in dict_like.items()}


def extend_int_to_list(int_like, target_num):
    # if int_like is None:
    #     logging.warning(f'Received None for expected int or list to extend to {target_num}.')
    if type(int_like) == int:
        int_like = [int_like] * target_num
    elif type(int_like) == list and len(int_like) == 1:
        int_like = int_like * target_num
    return int_like


def flatten_nested_dict(dict_obj: dict):
    flat_dict = {}
    for _, subdict in dict_obj.items():
        if type(subdict) == dict:
            flat_dict.update(subdict)
    return flat_dict


def flatten_listlike(listlike, safe=False):
    if safe:
        flat_list = []
        for l in listlike:
            if hasattr(l, '__iter__') and type(l) != str and l:
                flat_list.extend(l)
            # if type(l) == tuple or type(l) == list:
            #     flat_list.extend(l)
            else:
                flat_list.append(l)
        return flat_list
    else:
        return [item for sublist in listlike for item in sublist]


def flatten_nested_listlike(listlike):
    for v in listlike:
        if type(v) == list:
            listlike = flatten_listlike(listlike)
            listlike = flatten_nested_listlike(listlike)
            return listlike
    return listlike


def find_sublist_max(list_like):
    list_list_sizes = [len(l) for l in list_like if type(l) == list]
    return max(list_list_sizes)


def get_bulkiest_dict_key(dict_like):
    k_bulkiest = list(dict_like.keys())[0]
    prev_v = dict_like[k_bulkiest]
    for k, v in dict_like.items():
        if type(v) == list:
            if len(v) > len(prev_v):
                k_bulkiest = k
    return k_bulkiest


def get_first_elements(tuples_list: List[Union[tuple, list]], empty_replacement = ()):
    """ GC """
    return [[tt[0] for tt in t] if t else empty_replacement for t in tuples_list]


def get_unique(list_like: list):
    return list(set(list_like))


def inverse_dict(dict_like):
    return {v: k for k, v in dict_like.items()}


def make_attribute_list(typed_list, source_type, target_attribute):
    tlist = []
    for v in typed_list:
        if type(v) == source_type:
            tlist.append(getattr(v, target_attribute))
        if type(v) == list:
            tlist.append(make_attribute_list(v, source_type, target_attribute))
    return tlist


def merge_dicts(*dict_objs):
    all_dicts = {}
    for dict_obj in dict_objs:
        if type(dict_obj) == dict:
            all_dicts = {**all_dicts, **dict_obj}
            for k, v in dict_obj.items():
                if type(v) == dict and type(all_dicts[k]) == dict:
                    all_dicts[k] = merge_dicts(all_dicts[k], v)
                elif type(v) == dict and not type(all_dicts[k]) == dict:
                    logging.warning(
                        f'Could not merge {all_dicts[k]} with {v} for dict {dict_obj}')
        else:
            logging.warning(
                f'Could not merge object {dict_obj} of type {type(dict_obj)} with {all_dicts}')
    return all_dicts


def nest_list_dict(dict_of_lists: dict) -> list:
    vs = list(dict_of_lists.values())
    return [{k: vs[i][j] for i, k in enumerate(dict_of_lists.keys()) if vs[i]} for j in range(len(vs[0]))]


def replace_value(og_d: dict, key, new_val) -> dict:
    d = deepcopy(og_d)
    for k, v in d.items():
        if type(v) == dict:
            d[k] = replace_value(v, key, new_val)
    if key in d:
        d[key] = new_val
    return d


def rm_nones(list_like: list) -> list:
    return list(filter(lambda item: item is not None, list_like))
