import uuid
from .utils import write_pickle


class Experiment(object):

    def __init__(self, datasets_names, N_values, N_simulation, samples, parameters, save_path='experiments/'):

        self.datasets_names = datasets_names
        self.N_values = N_values
        self.N_simulation = N_simulation
        self.samples = samples
        self.parameters = parameters
        self.save_path = save_path
        self.identifier = uuid.uuid4()

    def save(self):
        write_pickle(self, path=self.save_path + str(self.identifier))

    def __str__(self):
        summarize = '------- Experiment {} -------- \n'.format(self.identifier)
        summarize += 'Datasets {} \n'.format(self.datasets_names)
        summarize += 'N_values tested: {}\n \n'.format(list(self.N_values))
        summarize += '------- Datasets Parameters -------- \n'
        for dataset_name in self.datasets_names:
            parameters = self.parameters[dataset_name]
            summarize += '- {} : \n'.format(dataset_name)
            summarize += 'N={}, d={} \n'.format(
                parameters['N'], parameters['d'])
            summarize += '{} \n'.format(parameters['parameters'])

        return summarize

    def __hash__(self):
        return hash(print(self.__str__()))


def get_datasets_parameters(datasets, datasets_names, parameters):
    datasets_parameters = {}
    for dataset, name, parameter in zip(datasets, datasets_names, parameters):
        parameters = {}
        parameters['N'] = dataset.shape[0]
        parameters['d'] = dataset.shape[1]
        parameters['parameters'] = parameter

        datasets_parameters[name] = parameters

    return datasets_parameters
