
# Copyright (c) 2023, Olivia Gallup
# All rights reserved.

# This source code is licensed under the MIT-style license found in the
# LICENSE file in the root directory of this source tree. 
    
import time


def time_it(func):
    def wrapper(*arg, **kw):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        # print("{0:.4f}s".format(t2 - t1), f"Result: {res}", f"\nFunction name: {func.__name__}")
        print("\n{0:.4f}s".format(t2 - t1), f"\nFunction name: {func.__name__}")
        return res
    return wrapper
