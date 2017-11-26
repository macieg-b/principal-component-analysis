import arff
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


class PCA:
    def __init__(self, data, result, percentage):
        self.data = data
        self.result = result
        self.classes = set(result)
        self.priority = {}
        self.percentage = percentage
        self.pca = 0

    def process_data(self):
        percentage_eigenvalue = {}
        data_transposed = np.array(self.data).transpose()
        cov = np.cov(data_transposed)
        eigenvalues, eigenvectors = np.linalg.eig(cov)

        sorted_eigenvalues = {i: list(eigenvalues).index(i) for i in sorted(eigenvalues, reverse=True)}
        self.priority = list(sorted_eigenvalues.values())

        eigenvalues_sum = sum(sorted_eigenvalues.keys())
        for i in sorted_eigenvalues:
            if sum(percentage_eigenvalue.keys()) < self.percentage:
                percentage_eigenvalue[i / eigenvalues_sum] = sorted_eigenvalues[i]

        self.pca = np.dot(data_transposed.transpose(), eigenvectors).transpose()
        return percentage_eigenvalue


class Chart:
    BLUE = "#7161ef"
    GREEN = "#00fa9a"
    RED = "#c8515f"

    def __init__(self):
        pass

    @staticmethod
    def two_dimensional(pca_object, ratio):
        for i in range(int(len(pca_object.result) * ratio)):
            if pca_object.result[i] == list(pca_object.classes)[0]:
                plt.scatter(pca_object.pca[pca_object.priority[0]][i], pca_object.pca[pca_object.priority[1]][i],
                            c=Chart.BLUE, s=4)
            elif pca_object.result[i] == list(pca_object.classes)[1]:
                plt.scatter(pca_object.pca[pca_object.priority[0]][i], pca_object.pca[pca_object.priority[1]][i],
                            c=Chart.GREEN, s=4)
            else:
                plt.scatter(pca_object.pca[pca_object.priority[0]][i], pca_object.pca[pca_object.priority[1]][i],
                            c=Chart.RED, s=4)
            plt.title("Rzut na dwie pierwsze skladowe")
            plt.xlabel('PCA-1')
            plt.ylabel('PCA-2')
        plt.show()

    @staticmethod
    def three_dimensional(pca_object, ratio):
        figure = plt.figure()
        axes_3d = Axes3D(figure)
        for i in range(int(len(pca_object.result) * ratio)):
            if pca_object.result[i] == list(pca_object.classes)[0]:
                axes_3d.scatter(pca_object.pca[pca_object.priority[0]][i], pca_object.pca[pca_object.priority[1]][i],
                                pca_object.pca[pca_object.priority[2]][i], c=Chart.BLUE, s=4)
            elif pca_object.result[i] == list(pca_object.classes)[1]:
                axes_3d.scatter(pca_object.pca[pca_object.priority[0]][i], pca_object.pca[pca_object.priority[1]][i],
                                pca_object.pca[pca_object.priority[2]][i], c=Chart.GREEN, s=4)
            else:
                axes_3d.scatter(pca_object.pca[pca_object.priority[0]][i], pca_object.pca[pca_object.priority[1]][i],
                                pca_object.pca[pca_object.priority[2]][i], c=Chart.RED, s=4)
            plt.title("Rzut na pierwszy trzy glowne skladowe")
            axes_3d.set_xlabel('PCA-1')
            axes_3d.set_ylabel('PCA-2')
            axes_3d.set_zlabel('PCA-3')
            axes_3d.zaxis.set_rotate_label(True)
            axes_3d.yaxis.set_rotate_label(True)
        plt.show()


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
