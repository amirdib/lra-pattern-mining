import pickle
import numpy as np
from time import time


def sample_supports(data, n):
    np.random.seed()  # important for parr computations
    N = data.shape[0]
    random_idx = np.random.choice(range(N), n, replace=False)
    sampled = data[random_idx, :]
    return sampled


def convert_numpy_to_sequences(array):
    N = array.shape[0]
    return [list(np.argwhere(array[i, :] == 1).reshape(-1)) for i in range(N)]


def write_numpy(array, path):
    data = convert_numpy_to_sequences(array)
    with open(path, 'w') as f:
        for datum in data:
            f.write(' '.join(map(str, datum)))
            f.write('\n')


def inverse_mapping(f):
    return f.__class__(map(reversed, f.items()))


def timeit(method):
    def timed(*args, **kw):
        tic = time()
        if 'n' in kw:
            print('Computing {} for n={}...'.format(method.__name__, n))
        else:
            print('Computing {} ...'.format(method.__name__))

        result = method(*args, **kw)
        toc = time()

        if 'n' in kw:
            print('{} Computed for n={} in {}s'.format(
                method.__name__, n, toc-tic))
        else:
            print('{} Computed in {}s'.format(method.__name__, toc-tic))
        return result
    return timed


def inverse_empirical_cdf(sample, alpha):
    ''' Compute the empirical inverse distribution for a sample and given alpha'''
    N = sample.shape[0]
    sorted_sample = np.sort(sample)
    return sorted_sample[int(N * (1-alpha))]


def read_pickle(path):
    with open(path, 'rb') as input_file:
        return pickle.load(input_file)


def write_pickle(data, path, **kwargs):
    with open(path, 'wb') as output_file:
        pickle.dump(data, output_file, protocol=4)


def read_json(path):
    with open(path, 'r') as input_file:
        return json.load(input_file)


def write_json(data, path):
    with open(path, 'w') as output_file:
        json.dump(data, output_file)

def plot2d(array, ax=None, **kwargs):
    ''' Two dimension plot of an array. 

    Args:
        array: 2D array. The right-most index is the dimension index.
        ax: Matplotlib Axes. If None, one would be created.
        kwargs: key word argument for the scatter plot.
    '''
    ax = plt.subplot() if ax is None else ax
    X = array[:, 0]
    Y = array[:, 1]
    ax.scatter(X, Y, **kwargs)

from time import time

def timer_func(func):
    def wrap_func(*args, **kwargs):
        t1 = time()
        result = func(*args, **kwargs)
        t2 = time()
        print(f'Function {func.__name__!r} executed in {(t2-t1):.4f}s')
        return result
    return wrap_func
