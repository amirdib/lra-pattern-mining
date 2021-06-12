import sys
sys.path.append('../')

import pandas as pd

from sklearn.preprocessing import MultiLabelBinarizer

from source.utils import sample_supports


class Dataset(object):
    def __init__(self):
        self.N = None
        self.D = None

    def sample(self,n):
        pass

class DatasetDataframe(Dataset):
    def __init__(self, path: str):
        super().__init__()

        def sample(self, n):
            pass
    

class DatasetFile(Dataset):
    def __init__(self, path: str):
        super().__init__()
        self.path = path
        self.name = path.split('/')[-1].replace('.txt', '')
        self.binarized_data = None
        self.classes = None
        self.sampled_data = None

    def _ingest(self):
        with open(self.path, 'r') as f:
            lines = f.readlines()
            self.data = [line.split(' ')[:-1] for line in lines]
        return self.data

    def binarize(self):
        data = self._ingest()
        mlb = MultiLabelBinarizer()
        self.binarized_data = mlb.fit_transform(data)
        self.classes = mlb.classes_
        self.mlb = mlb

    def get_binarized_data(self):
        return self.binarized_data

    def get_items(self):
        return self.classes

    def get_pattern_data(self):
        return self.data

    def get_dataset_features(self):
        self.N = self.binarized_data.shape[0]
        self.D = self.binarized_data.shape[1]
        return self.name, self.D, self.N

    #TODO: use random state
    def sample(self, n, random_state=32):
        if n is None:
            n = self.N
        else:
            self.n = n
        self.sampled_data = self.mlb.inverse_transform(
            sample_supports(self.binarized_data, n))
        return self.sampled_data

    def __call__(self):
        self.binarize()
        self.get_dataset_features()

    def __str__(self):
        return 'Dataset {} : N = {} D = {}'.format(self.name, self.N, self.D)
