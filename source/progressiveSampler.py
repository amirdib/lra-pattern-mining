import sys
sys.path.append('../')

from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor

import numpy as np
import pandas as pd
from tqdm.auto import tqdm

from source.datasets import Dataset
from source.spmf import SPMF, ItemsetMiner

class ProgressiveSampler(object):
    def __init__(self, dataset: Dataset, sample_sizes: list, args: str = .5):
        self.sample_sizes = sample_sizes
        self.dataset = dataset
        self.true_supports = None
        self.args = args
        self.verbose = False

    def compute_true_supports(self):
        data = self.dataset.sample(None)
        miner = ItemsetMiner(tdata=data, model='FPGrowth_itemsets',
                             args=self.args, verbose=self.verbose)
        miner.run(verbose=self.verbose)
        self.true_supports = miner.to_pandas_dataframe()

    def compute_sampled_supports(self, n: int):
        data = self.dataset.sample(n=n)
        miner = ItemsetMiner(tdata=data, model='FPGrowth_itemsets',
                             args=self.args, verbose=self.verbose)
        miner.run(verbose=self.verbose)
        sampled_supports = miner.to_pandas_dataframe()
        return sampled_supports

    def compute_multiple_sampled_supports(self):
        with ProcessPoolExecutor() as p:
            sampled_supports = p.map(
                self.compute_sampled_supports, self.sample_sizes)
        return self.compute_multiple_maximum_deviation(list(sampled_supports))

    def compute_deviation(self, true_supports: pd.DataFrame, sampled_supports: pd.DataFrame):
        support_comparaison = true_supports.merge(
            sampled_supports, on='pattern', suffixes=('_true', '_sampled'))
        support_comparaison['deviation'] = np.abs(
            support_comparaison['support_true'] - support_comparaison['support_sampled'])
        return support_comparaison

    def compute_multiple_deviation(self, sampled_supports: list):
        multiple_deviation = [self.compute_deviation(
            self.true_supports, supports) for supports in sampled_supports]
        return multiple_deviation

    def compute_multiple_maximum_deviation(self, sampled_supports: list):
        multiple_deviation = self.compute_multiple_deviation(
            sampled_supports=sampled_supports)
        return [support_comparaison['deviation'].max() for support_comparaison in multiple_deviation]

    def run(self, n_runs):
        self.compute_true_supports()
        return np.array([self.compute_multiple_sampled_supports() for _ in tqdm(range(0, n_runs))])

    def plot(self):
        pass
