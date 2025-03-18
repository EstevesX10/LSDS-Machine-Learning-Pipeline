from time import (time)

def timeit(func):
    def wrapper(self, *args, **kwargs):
        start = time()
        output = func(self, *args, **kwargs)
        return (output, time() - start)
    return wrapper