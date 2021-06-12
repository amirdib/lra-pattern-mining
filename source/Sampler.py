import plotly.express as px
import copy 

import sys
sys.path.append('../')
from pathlib import Path
BASE_PATH = str(Path.cwd().parent)
from itertools import product

from sklearn.preprocessing import MultiLabelBinarizer
import matplotlib as mpl
import seaborn as sns
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
from source.datasets import DatasetFile, Dataset
#from source.distribution import make_random_noise
mpl.rcParams['figure.figsize'] = 15, 15
mpl.rcParams['font.family'] = 'serif'
mpl.rcParams['axes.titlesize'] = 24
mpl.rcParams['axes.labelsize'] = 20
mpl.rcParams['xtick.labelsize'] = 15
mpl.rcParams['ytick.labelsize'] = 15
mpl.rcParams['legend.fontsize'] = 16

from source.datasets import Dataset
from source.spmf import SPMF, ItemsetMiner
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from subprocess import Popen,PIPE,STDOUT

class DatasetDataframe(Dataset):
    def __init__(self, dataframe, name):
        super().__init__()
        self.binarized_data = dataframe
        self.name = name
        self.N = dataframe.shape[0]
        self.d = dataframe.shape[1]
        self.dataset = None
        
    def convert_to_transactions(self,dataset=None):
        if dataset is None:
            dataset = self.binarized_data
        mlb = MultiLabelBinarizer()
        mlb.classes_ = np.array(list(list(range(1,self.d+1)))).astype(str)
        self.dataset = mlb.inverse_transform(dataset.values)
        return self.dataset
        
    def sample(self, n=None, random_state=None):
        if n is None:
            n = self.N
        else:
            self.n = n
        self.sampled_data = self.convert_to_transactions(self.binarized_data.sample(n,random_state=random_state))
        return self.sampled_data
    
    def __call__(self):
        self.convert_to_transactions()
    
    def write_data_labels(self,path):
        with open(path + '{}.dat'.format(self.name),'w') as fp:
            '''Write the transactions on the file path fp'''
            for i, seq in enumerate(self.dataset):
                if seq == ():
                    line = '0 \n'
                else:
                    line = ' '.join(list(map(lambda x: str(x), seq))) + '\n'
                    
                fp.write(str(line))
            fp.close()

class ProgressiveSamplerMCR(object):
    def __init__(self, sample_size, dbname,m=1000):
        self.sample_size = sample_size
        self.m = m
        self.epsilon = None
        self.dbname = dbname
        self.mcrapper_path = BASE_PATH + '/external/MCRapper/'

    def run_mcrapper(self):
        
        
        cwd_path = self.mcrapper_path + 'scripts/'       
        script_path = cwd_path + 'run_mcrapper.py'
        
        process = Popen(['python',script_path,
                             '-db', self.dbname, '-sz', str(self.sample_size), '-j', str(self.m)], stdout=PIPE, stderr=STDOUT, cwd = cwd_path)
        while True:
            nextline = process.stdout.readline()
            if nextline == b'' and process.poll() is not None:
                break
            sys.stdout.write(nextline.decode('utf-8'))
            sys.stdout.flush()
        print('Process Ended')
        
        output = process.communicate()[0]
        exitCode = process.returncode
        
        if (exitCode == 0):
            print(output.decode("utf-8"))
        else:
            print('Error. Process exited with {} code. \n'.format(exitCode))
            print(output.decode("utf-8"))
            
    def get_results(self):
        result_path = self.mcrapper_path +  'scripts/results_radest.csv'
        header = ['database', 'sample_size', 'delta','col1',
                  'col2','t_correction','epsilon_n1',
                  'epsilon','rademacher_average',
                  'omega1','rho1','n_items_explored',
                  'col5','epsilon_var','epsilon_hyb','u1','u2','u3']
        return pd.read_csv(result_path,header=None, sep=';',names=header)

class ProgressiveSamplerMCR(object):
    def __init__(self, dbname, sample_size, delta=.1, m=1000):
        self.sample_size = sample_size
        self.m = m
        self.delta = delta
        self.epsilon = None
        self.dbname = dbname
        self.mcrapper_path = BASE_PATH + '/external/MCRapper/'

    def run_mcrapper(self):
        
        if self.check_if_experiment_exists():
            cwd_path = self.mcrapper_path + 'scripts/'       
            script_path = cwd_path + 'run_mcrapper.py'
            
            process = Popen(['python',script_path,'--delta', str(self.delta),
                                 '-db', self.dbname, '-sz', str(self.sample_size), '-j', str(self.m)], stdout=PIPE, stderr=STDOUT, cwd = cwd_path)
            while True:
                nextline = process.stdout.readline()
                if nextline == b'' and process.poll() is not None:
                    break
                sys.stdout.write(nextline.decode('utf-8'))
                sys.stdout.flush()
            print('Process Ended')
            
            output = process.communicate()[0]
            exitCode = process.returncode
            
            if (exitCode == 0):
                print(output.decode("utf-8"))
            else:
                print('Error. Process exited with {} code. \n'.format(exitCode))
                print(output.decode("utf-8"))
            
    def get_all_results(self):
        result_path = self.mcrapper_path +  'scripts/results_radest.csv'
        header = ['database', 'sample_size', 'delta','col1',
                  'col2','t_correction','epsilon_n1',
                  'epsilon','rademacher_average',
                  'omega1','rho1','n_items_explored',
                  'col5','epsilon_var','epsilon_hyb','u1','u2','u3']
        return pd.read_csv(result_path,header=None, sep=';',names=header)
    
    def get_result(self):
        return self.get_all_results().query(''' sample_size == @self.sample_size and database == @self.dbname  and delta == @self.delta   ''')
        return pd.read_csv(result_path,header=None, sep=';',names=header)
    def check_if_experiment_exists(self):
        return self.get_result().empty
        
    
        
class ProgressiveSampler(object):
    def __init__(self, dataset: Dataset, sample_sizes: list, args: str = .5,n_runs: int = 1,model = 'FPGrowth_itemsets'):
        self.sample_sizes = sample_sizes
        self.dataset = dataset
        self.true_supports = None
        self.args = args
        self.verbose = False
        self.n_runs = n_runs
        self.model = model
        #self.model = 'FPClose'
        self.records = {} #hierarchical Dictionnary of dataframes

    def compute_true_supports(self):
        data = self.dataset.sample(None)
        miner = ItemsetMiner(tdata=data, model=self.model,
                             args=self.args, verbose=self.verbose)
        miner.run(verbose=self.verbose)
        self.true_supports = miner.to_pandas_dataframe()

    def compute_sampled_supports(self, n: int):
        data = self.dataset.sample(n=n,random_state=np.random.RandomState())
        miner = ItemsetMiner(tdata=data, model=self.model,
                             args=self.args, verbose=self.verbose)
        miner.run(verbose=self.verbose)
        sampled_supports = miner.to_pandas_dataframe()
        return sampled_supports

    def compute_multiple_sampled_supports(self, n_run):
        with ProcessPoolExecutor() as p:
            multiple_sampled_supports = p.map(
                self.compute_sampled_supports, self.sample_sizes)
        
        multiple_sampled_supports = list(multiple_sampled_supports)
        #self.multiple_sampled_supports = multiple_sampled_supports
        self.records['r{}'.format(n_run +1)] = {sample_size:sampled_supports for sample_size,sampled_supports in zip(self.sample_sizes,multiple_sampled_supports)}
        return self.compute_multiple_maximum_deviation(list(multiple_sampled_supports))

    def compute_deviation(self, true_supports: pd.DataFrame, sampled_supports: pd.DataFrame):
        support_comparaison = pd.merge(true_supports, sampled_supports, 
                                       on='pattern', 
                                       suffixes=('_true', '_sampled'), how='outer')
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

    def run(self):
            
        self.compute_true_supports()
        self.records['r{}'.format(self.dataset.N)] = self.true_supports
        return np.array([self.compute_multiple_sampled_supports(n_run) for n_run in tqdm(range(0, self.n_runs))])

    def plot(self):
        pass
    
class Records(object):
    def __init__(self,records, sample_sizes, N, d, n_runs):
        self.records = copy.deepcopy(records)
        self.sample_sizes = sample_sizes
        self.N = N
        self.d = d
        self.n_runs = n_runs
        
    def extract_global_index(self):
        r_range = ['r{}'.format(n_run) for n_run in range(1,self.n_runs+1)]
        #index = [self.records[n_run][sample_size]['pattern'].to_list() for n_run, sample_size in product(r_range, sample_sizes)]
        index = []
        for n_run, sample_size in product(r_range, self.sample_sizes):
            index += self.records[n_run][sample_size]['pattern'].to_list()
        index += self.records['r{}'.format(self.N)]['pattern'].to_list()
        index = list(set(index))
        self.index = index
        return index
        
    def global_reindex(self):
        r_range = ['r{}'.format(n_run) for n_run in range(1,self.n_runs+1)]
        for n_run, sample_size in product(r_range, self.sample_sizes):
            #print('Run {} with N={} contain {} patterns'.format(n_run,sample_size,self.records[n_run][sample_size].shape[0]))
            self.records[n_run][sample_size] = self.records[n_run][sample_size].set_index('pattern').reindex(self.index)
            
        self.records['r{}'.format(self.N)] = self.records['r{}'.format(self.N)].set_index('pattern').reindex(self.index)
        #print('Run full DB contain {} patterns'.format(self.records['r0'].shape[0]))
    
    def to_dataframe(self):
        r_range = ['r{}'.format(n_run) for n_run in range(1,self.n_runs+1)]
        col_index = pd.MultiIndex.from_product([[ 'r{}'.format(n_run) for n_run in list(range(1,self.n_runs+1))], 
                                        [self.N] + self.sample_sizes])

        dataframe = pd.DataFrame( index=self.index,columns = col_index)
        dataframe = dataframe.swaplevel(axis=1)
        dataframe
        
        
        dataframe[self.N] = self.records['r{}'.format(self.N)]['support']
        
        for n_run, sample_size in product(r_range, self.sample_sizes):
            dataframe[(sample_size,n_run)] = self.records[n_run][sample_size]['support']
            
        dataframe = dataframe.swaplevel(axis=1).sort_index()
        dataframe['length'] = dataframe.index
        
        dataframe['length'] = dataframe['length'].apply(lambda x: len(x))
        dataframe = dataframe.sort_values('length', ascending=True)
        
        dataframe = dataframe.fillna(0) #TODO Check !
        
        dataframe = dataframe.swaplevel(axis=1).sort_index()
        
        self.dataframe = dataframe
        return self.dataframe
    
    def __call__(self):
        self.extract_global_index()
        self.global_reindex()
        return self.to_dataframe()
    
    def plot(self):
        df_plot = self.dataframe.groupby(axis=1,level=0).mean()
        df_plot.columns = self.sample_sizes + [self.N] + ['length']
        df_plot = pd.melt(df_plot.reset_index(),['length', 'index'])
        df_plot
        
        fig = px.line(df_plot, x="variable", y="value", color="length",
                      line_group="index", hover_name="index")
        fig.show()
    
