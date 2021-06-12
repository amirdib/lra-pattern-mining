import sys
sys.path.append('../')


import subprocess
import shlex
import tempfile

import numpy as np
import pandas as pd

from source.utils import read_json, inverse_mapping

from pathlib import Path

TMP_PATH = str(Path.cwd().parent / 'tmp')
SPMF_PATH = str(Path.cwd().parent / 'external') + '/'


class SPMFParserError(Exception):
    pass


def preprocess_to_tdatabase(preprocess):
    N = len(preprocess.iloc[1, 1:].index.tolist())
    dic_codes = {code: idx for code, idx in enumerate(
        preprocess.iloc[1, 1:].index.tolist())}
    datah = preprocess[preprocess['fonction']
                       == 0].iloc[:, 1:].values
    dataf = preprocess[preprocess['fonction']
                       != 0].iloc[:, 1:].values

    return get_sequences(datah, col_index=dic_codes), get_sequences(dataf, col_index=dic_codes)


def get_sequences(data, col_index):
    a = np.argwhere(data > 0)
    n = np.unique(a[:, 0])
    seqs = [list(a[a[:, 0] == i, 1]) for i in n]
    return [[col_index[e] for e in seq] for seq in seqs]


class PatternMiner(object):
    def __init__(self, tdata, model, args):
        self.tdata = tdata
        self.database_length = len(tdata)
        self.model = model
        self.args = args
        self.output = None

    def run(self, verbose=True):

        input_tempfile = tempfile.NamedTemporaryFile(
            mode='w',  suffix='_input', dir=TMP_PATH, delete=False)
        # print(input_tempfile)
        output_tempfile = tempfile.NamedTemporaryFile(
            mode='r', suffix='_output', dir=TMP_PATH, delete=False)

        try:
            self._write_transactions(self.tdata, input_tempfile)

            model = SPMF(input_path=input_tempfile.name,
                         output_path=output_tempfile.name,
                         model=self.model,
                         args=self.args)
            stdout = model.run_cmd()
            if verbose is True:
                print(stdout)
            self.output = output_tempfile.read()
            return self.output

        finally:
            # Not an optionnal closing: garbage collector automatically close file path when counter reach zero only in Cpython.
            input_tempfile.close()
            output_tempfile.close()

    def _save(self, output_file):
        with open(output_file, 'r') as f:
            return f.read()

    def to_pandas_dataframe(self):
        result_patterns = []
        for result in self.output.split('\n')[:-1]:
            pattern = result.split(' #')[0].split(' ')
            pattern = tuple(map(int, pattern))
            support = result.split('#SUP: ')[1]

            result_patterns.append(
                {'pattern': pattern, 'support': int(support)})
        result_dataframe = pd.DataFrame(result_patterns)
        result_dataframe['support'] /= self.database_length
        return result_dataframe


class GoKrimp(PatternMiner):

    def __init__(self, tdata, args, code_label, **kwargs):
        super().__init__(tdata=tdata, model='GoKrimp', args=args)
        self.code_label = code_label

    def _write_transactions(self, transactions, fp):
        '''Write the transactions on the file path fp'''
        for i, seq in enumerate(transactions):
            line = ' -1 '.join([str(self.code_label[e])
                                for e in seq]) + ' -2\n'
            fp.write(line)
        fp.close()


class ItemsetMiner(PatternMiner):

    def __init__(self, tdata, args, model, **kwargs):
        super().__init__(tdata=tdata, model=model, args=args)

    def _write_transactions(self, transactions, fp):
        '''Write the transactions on the file path fp'''
        for i, seq in enumerate(transactions):
            line = ' '.join([str(e) for e in seq]) + '\n'
            fp.write(line)
        fp.close()


class SPMF(object):
    '''Interface used to run SPMF algorithms'''

    def __init__(self, input_path, output_path, model, args, verbose=True):
        self.args = args

        self.input_path = input_path
        self.output_path = output_path 

        self.model = model

        self.verbose = verbose

    def run_cmd(self):
        cmd = 'java -jar {}spmf.jar run {} {} {} {}'.format(
            SPMF_PATH, self.model, self.input_path, self.output_path, self.args)
        # if self.verbose is True:
        #     print(cmd)
        args = shlex.split(cmd)
        process = subprocess.Popen(
            args,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE)
        out, err = process.communicate()
        if err:
            raise SPMFParserError(err.decode('utf-8'))

        return out.decode('utf-8')
