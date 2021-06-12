from time import time

from pathlib import Path

from datasets import Dataset
from progressiveSampler import ProgressiveSampler

DIR_PATH = Path.cwd().parent

if __name__ == '__main__':
    datasets_names = ['accidents', 'chess',
                      'connect', 'pumsb', 'retail', 'mushrooms']
    datasets_names = ['accidents', 'chess']
    datasets_path = [str(DIR_PATH / 'data' / '{}.txt'.format(name))
                     for name in datasets_names]

    start = time()
    chess = Dataset(path=datasets_path[1])
    chess()

    progressive_sampling = ProgressiveSampler(dataset=chess, sample_sizes=[10,100, 1000], args=.7)
    deviations = progressive_sampling.run(n_runs = 10)
    print(deviations)
    end = time()
    print('Execution took {} s'.format(int(end-start)))
