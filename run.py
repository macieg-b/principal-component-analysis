from model import FileLoader
from model import PCA
from model import Chart


# FILE_PATH_ONE = "data/iris.arff"
# data_iris, y_iris = FileLoader.load_data(FILE_PATH_ONE)
# pca_iris = PCA(data_iris, y_iris, 0.9)
# iris_result = pca_iris.process_data()
# print(iris_result)
# Chart.two_dimensional(pca_iris)
# Chart.three_dimensional(pca_iris)

FILE_PATH = "data/waveform5000.arff"
data, y = FileLoader.load_data(FILE_PATH)
pca = PCA(data, y, 0.9)
waveform_result = pca.process_data()
print(waveform_result)
Chart.two_dimensional(pca)
Chart.three_dimensional(pca)
