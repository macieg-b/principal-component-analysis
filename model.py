import arff
import numpy as np


class PCA:

    def __init__(self, data, result):
        self.data = data
        self.result = result
        self.classes = set(result)

    def process_data(self):
        x = np.array(self.data).transpose()
        cov = np.cov(x)
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        a = 5


class FileLoader:

    def __init__(self):
        pass

    @staticmethod
    def load_data(file_path):
        data_set = arff.load(open(file_path, 'rb'))
        data = data_set['data']
        probe_length = len(data[0])
        classes = []

        for vector in data:
            classes.append(vector[probe_length - 1])
            del vector[probe_length - 1]
        return data, classes
