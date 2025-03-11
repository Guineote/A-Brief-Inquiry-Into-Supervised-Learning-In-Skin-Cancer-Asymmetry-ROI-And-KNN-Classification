import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

histogram_vectors = np.load('histogram_vectors.npy')
categories = pd.read_csv('HAM10000_metadata.csv').iloc[:, 2]

pca = PCA(n_components=2)

histogram_vectors_pca = pca.fit_transform(histogram_vectors)

plt.figure(figsize=(10, 8))

unique_categories = categories.unique()
colors = ['blue', 'green', 'red', 'cyan', 'magenta', 'yellow', 'black', 'orange', 'purple', 'brown']

for i, category in enumerate(unique_categories):
    mask = (categories == category)
    plt.scatter(histogram_vectors_pca[mask, 0], histogram_vectors_pca[mask, 1], 
                color=colors[i], label=category, alpha=0.5)


plt.title('PCA de Vectores de Histograma')
plt.xlabel('Componente Principal 1')
plt.ylabel('Componente Principal 2')
plt.legend()

plt.show()
