import numpy as np
from sklearn.decomposition import PCA


histogram_vectors = np.load('histogram_vectors.npy')

pca = PCA(n_components=2)

histogram_vectors_pca = pca.fit_transform(histogram_vectors)

np.save('histogram_vectors_pca.npy', histogram_vectors_pca)


print(histogram_vectors_pca.shape)  # Debería imprimir (10015, 2)
