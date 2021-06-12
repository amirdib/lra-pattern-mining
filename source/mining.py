from pathlib import Path
from collections import OrderedDict
import uuid
from time import time
from sklearn.preprocessing import MultiLabelBinarizer
import pandas as pd
from .utils import sample_supports, write_numpy
from .spmf import SPMF

from source.datasets import DatasetFile, Dataset
from source.Sampler import DatasetDataframe,ProgressiveSampler

Path.cwd().parent / 'experiments'


def get_spmf_supports(n, data, arg, model='FPGrowth_itemsets', verbose=True):

    tic = time()

    N = data.shape[0]
    if n is None:
        n = N
    if verbose is True:
        print('Computing for n={}...'.format(n))
    data = sample_supports(data=data, n=n)
    dataset_path = 'data/progressive_sampling{}_{}.txt'.format(n, uuid.uuid4())
    write_numpy(data, path=dataset_path)
    res = get_support(dataset_path, model=model, arg=arg, verbose=False)
    supports = normalize_support(res, N=n)
    toc = time()
    if verbose is True:
        print('Supports computed for n={} in {}s'.format(n, toc-tic))
    return supports


def get_support(dataset_path, model='FPGrowth_itemsets', arg='.8', verbose=False):

    output_path = dataset_path + '_output.txt'
    fp = SPMF(input_path=dataset_path, output_path=output_path,
              model=model, args=arg, verbose=verbose)
    output = fp.run_cmd()
    if verbose is True:
        print(output)

    with open(output_path, 'r') as fp:
        res = read_transactions(fp)

    return res


def normalize_support(result_dict, N):
    normalized_result_dict = {key: value /
                              N for key, value in result_dict.items()}
    return dict(sorted(normalized_result_dict.items(), key=lambda k: k[1], reverse=True))


def read_transactions(fp):
    ''' Reads a transaction from SPMF algorithm'''
    items = dict()
    for line in fp.readlines():
        support = float(line.split('#SUP')[1].split(' ')[-1].replace('\n', ''))
        item = line.split('#SUP')[0].split(' ')[:-1]
        item = tuple(int(e) for e in item)
        items[tuple(item)] = support
    return items


def normalize_support(result_dict, N):
    normalized_result_dict = {key: value /
                              N for key, value in result_dict.items()}
    return dict(sorted(normalized_result_dict.items(), key=lambda k: k[1], reverse=True))


def supports_to_df(supports):
    coord = list(map(lambda x: (len(x[0]), x[0], x[1]), supports.items()))
    return pd.DataFrame(coord, columns=['len', 'item', 'support'])


def ingest(path):
    with open(path, 'r') as f:
        lines = f.readlines()
        data = [line.split(' ')[:-1] for line in lines]
    return data


def transform_to_binary(pattern, D, classes):
    x = np.zeros(D,)
    for item in pattern:
        x[np.where(classes == item)] = 1
    return x


def preprocessing(data):
    mlb = MultiLabelBinarizer()
    binarized_data = mlb.fit_transform(data)
    classes = mlb.classes_.astype(int)
    return binarized_data, classes


def preprocess(path):
    data = ingest(path)
    binarized_data, classes = preprocessing(data)
    return binarized_data, classes

def generate_itemsets(d):
    item_dictionnary = np.array([letter for letter in string.ascii_lowercase[:d]])
    return np.array([transform_to_binary(pattern, d=d, classes=item_dictionnary) 
          for j in np.arange(d+1)
          for pattern in combinations(item_dictionnary, j)
             ])

def isin_pattern(x,y):
    return int((y <= x).sum() == x.shape[0])

def print_wrt_F(q_list):
    for f, support in zip(itemsets[1:],q_list):
        str_f = ''.join([str(item) for item in f])
        print('length:{}, f={}: {}'.format(f.sum(),str_f,support))


def binarize_itemsets(transactions, mlb=None):
    if mlb is None:
        mlb = MultiLabelBinarizer()
    #print(len(mlb.classes_))
    return mlb.transform(transactions)


def mine_closed_itemset(data, sample_size=None, args=.01):
    # if not isinstance(data ,DatasetFile):
    #     data = DatasetDataframe(dataframe=data, name='')

    progressive_sampling = ProgressiveSampler(dataset=data, sample_sizes=None, args=args,n_runs=1, model='FPClose')
    progressive_sampling_dataframe = progressive_sampling.compute_sampled_supports(n=sample_size)
#    progressive_sampling_dataframe['pattern'] = progressive_sampling_dataframe['pattern'].apply(lambda itemset: list(map(lambda item: item-1,itemset)))
    return progressive_sampling_dataframe


def get_closed_itemsets(dataframe, sample_size=None, args=.01):
    closed_itemsets_dataframe = mine_closed_itemset(data=dataframe,sample_size=sample_size, args=args)
    closed_itemsets = list(closed_itemsets_dataframe['pattern'].values)
    return closed_itemsets

def compute_true_average(data,itemset):
    #binary_retail[:,9693].sum()/binary_retail.shape[0] Sanity Check
    N = data.shape[0]
    return np.sum(vect_isin(data,itemset))/N


def compute_empirical_support(data,itemset):
    return vect_isin(data, itemset).sum()

def compute_empirical_supports(data,itemsets):
    return np.array([vect_isin(data, itemset).sum() for itemset in tqdm(itemsets)])/data.shape[0]

def index_to_binary_transaction(indexes,dimension):
    itemset = np.zeros(dimension)
    itemset[list(indexes)] = int(1)
    return itemset
