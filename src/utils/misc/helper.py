

def next_wrapper(generator):
    return next(generator)


def none_func(input):
    return None


def vanilla_return(input):
    return input


def processor(input, funcs):
    for func in funcs:
        input = func(input)
    return input
