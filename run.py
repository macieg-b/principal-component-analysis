from model import DataLoader
from model import PCA
from model import Chart


FILE_PATH_ONE = "data/iris.arff"
data_iris, y_iris = DataLoader.load_data(FILE_PATH_ONE)
attribute_1, attribute_2, attribute_3 = DataLoader.get_random_attributes(data_iris)
pca_iris = PCA(data_iris, y_iris, 0.9)
iris_eigenvalues, iris_result = pca_iris.process_data()
print("\n\n")
print("File - " + FILE_PATH_ONE)
print("Sorted eigenvalues (descending):")
print(iris_eigenvalues)
print("0.9:")
print(iris_result)
Chart.two_dimensional_random(attribute_1, attribute_2)
Chart.three_dimensional_random(attribute_1, attribute_2, attribute_3, y_iris)
Chart.two_dimensional(pca_iris)
Chart.three_dimensional(pca_iris)

FILE_PATH = "data/waveform5000.arff"
data, y = DataLoader.load_data(FILE_PATH)
attribute_1, attribute_2, attribute_3 = DataLoader.get_random_attributes(data)
pca = PCA(data, y, 0.9)
waveform_eigenvalues, waveform_result = pca.process_data()
print("\n\n")
print("File - " + FILE_PATH)
print("Sorted eigenvalues (descending):")
print(waveform_eigenvalues)
print("0.9:")
print(waveform_result)
Chart.two_dimensional_random(attribute_1, attribute_2)
Chart.three_dimensional_random(attribute_1, attribute_2, attribute_3, y)
Chart.two_dimensional(pca)
Chart.three_dimensional(pca)
