
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    


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
