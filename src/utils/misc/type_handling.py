

def merge_dicts(*dict_objs):
    all_dicts = {}
    for dict_obj in dict_objs:
        if type(dict_obj) == dict:
            all_dicts = {**all_dicts, **dict_obj}
    return all_dicts


def flatten_nested_dict(dict_obj):
    flat_dict = {}
    for _, subdict in dict_obj:
        if type(subdict) == dict:
            flat_dict.update(subdict)
    return flat_dict


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


def assert_uniform_type(list_like, target_type):
    assert all(type(list_like) == target_type)
