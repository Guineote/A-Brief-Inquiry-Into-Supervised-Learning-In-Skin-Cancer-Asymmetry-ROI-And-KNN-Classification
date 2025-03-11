import numpy as np
import pandas as pd
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("Análisis de silueta para lesiones cutáneas")
print("Este script realiza un análisis de silueta y visualización de clusters")
print("para diferentes tipos de lesiones cutáneas basado en histogramas de vectores.")
print("\n1. Cargando datos...")
X = np.load('histogram_vectors.npy', allow_pickle=True)
labels_df = pd.read_csv('HAM10000_metadata.csv')  
labels = labels_df['dx'].values  
print(f"Datos cargados: {X.shape}")
print(f"Número de etiquetas: {len(labels)}")
print("\n2. Preprocesando datos...")

if not isinstance(X, np.ndarray):
    X = np.array(X)
if X.ndim > 2:
    X = X.reshape(X.shape[0], -1)
    
print(f"Forma procesada: {X.shape}")
print("\n3. Realizando análisis de silueta...")
unique_labels = np.unique(labels)
n_clusters = len(unique_labels)
fig, (ax1, ax2) = plt.subplots(1, 2)
fig.set_size_inches(18, 7)
ax1.set_xlim([-0.5, .5])
ax1.set_ylim([0, len(X) + (n_clusters + 1) * 10])
silhouette_avg = silhouette_score(X, labels)
print(f"Puntuación de silueta promedio: {silhouette_avg:.3f}")
sample_silhouette_values = silhouette_samples(X, labels)
y_lower = 10

for i, cluster in enumerate(unique_labels):
    ith_cluster_silhouette_values = sample_silhouette_values[labels == cluster]
    ith_cluster_silhouette_values.sort()
    size_cluster_i = ith_cluster_silhouette_values.shape[0]
    y_upper = y_lower + size_cluster_i
    color = cm.nipy_spectral(float(i) / n_clusters)
    ax1.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_cluster_silhouette_values,
                      facecolor=color, edgecolor=color, alpha=0.7)
    ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(cluster))
    y_lower = y_upper + 10

ax1.set_title("Silhouett graph")
ax1.set_xlabel("Coefficient values of the silhouette")
ax1.set_ylabel("Type of injury")
ax1.axvline(x=silhouette_avg, color="red", linestyle="--")
ax1.set_yticks([])
ax1.set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
print("\n4. Visualizando clusters...")
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X)
colors = cm.nipy_spectral(np.arange(float(n_clusters)) / n_clusters)

for cluster, color in zip(unique_labels, colors):
    mask = labels == cluster
    ax2.scatter(X_pca[mask, 0], X_pca[mask, 1], marker=".", s=30, lw=0, alpha=0.7, 
                c=[color], edgecolor="k", label=cluster)

ax2.legend(scatterpoints=1, loc='center left', bbox_to_anchor=(1, 0.5))
ax2.set_title("Visualización de los datos agrupados")
ax2.set_xlabel("First Principal Component")
ax2.set_ylabel("Second Principal Component")
plt.suptitle("Silhouette Analysis", fontsize=14, fontweight="bold")
print("\n5. Guardando resultados...")
plt.tight_layout()
plt.savefig('silhouette_analysis_skin_lesions.png')
plt.show()
print("\n6. Calculando métricas de validación adicionales...")
X_train, X_test, y_train, y_test = train_test_split(X, labels, test_size=0.2, random_state=42)
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
print("\nInforme de clasificación:")
print(classification_report(y_test, y_pred))
print("\nMatriz de confusión:")
cm = confusion_matrix(y_test, y_pred)
print(cm)
print("\nAnálisis completado.")