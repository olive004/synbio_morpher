

def remove_special_py_functions(string_list: list) -> list:
    return [s for s in string_list if '__' not in s]
