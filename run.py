from model import FileLoader
from model import PCA


FILE_PATH = "data/iris.arff"


data, y = FileLoader.load_data(FILE_PATH)
pca = PCA(data, y)
pca.process_data()