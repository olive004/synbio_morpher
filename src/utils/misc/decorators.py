import time


def time_it(func):
    def wrapper(*arg, **kw):
        '''source: http://www.daniweb.com/code/snippet368.html'''
        t1 = time.time()
        res = func(*arg, **kw)
        t2 = time.time()
        print("{0:.4f}s".format(t2 - t1), f"Result: {res}", f"\nFunction name: {func.__name__}")
    return wrapper
