import functools
import time

from torch import Tensor


def timeit(func):
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        start_time = time.time()
        result = func(*args, **kwargs)
        elapsed_time = time.time() - start_time
        print('function [{}] finished in {} ms'.format(
            func.__name__, int(elapsed_time * 1_000)))
        return result

    return new_func


def print_size(tensor: Tensor, name: str) -> None:
    print(f'{name} size: {tensor.size()}')
