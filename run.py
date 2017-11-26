from model import FileLoader
from model import PCA
from model import Chart


FILE_PATH = "data/iris.arff"
data, y = FileLoader.load_data(FILE_PATH)
pca = PCA(data, y, 0.9)
pca.process_data()
Chart.two_dimensional(pca, 1)
Chart.three_dimensional(pca, 1)

FILE_PATH = "data/waveform5000.arff"
data, y = FileLoader.load_data(FILE_PATH)
pca = PCA(data, y, 0.9)
pca.process_data()
Chart.two_dimensional(pca, 1)
Chart.three_dimensional(pca, 1)