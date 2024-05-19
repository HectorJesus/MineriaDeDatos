#El clustering agrupa un conjunto de objetos de tal 
# manera que los objetos en el mismo grupo 
# (llamado clúster) son más similares entre sí que 
# aquellos en otros grupos. 

#K-means

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# Cargar el dataset
data = pd.read_csv('powerLiftDataClean.csv')

# Mostrar las primeras filas del dataset
print(data.head())

# Seleccionar características para el clustering
features = data[['BestSquatKg', 'BestDeadliftKg']]

# Normalizar las características
scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

# Aplicar K-means con k=2
kmeans = KMeans(n_clusters=2, random_state=42)
kmeans.fit(scaled_features)

# Obtener etiquetas de clústeres
labels = kmeans.labels_

# Agregar etiquetas al dataset original
data['Cluster'] = labels

# Mostrar las primeras filas con etiquetas de clústeres
print(data.head())

# Colores para los clústeres
colors = ['red', 'blue']

# Graficar los datos agrupados
plt.figure(figsize=(10, 8))
for i in range(2):
    cluster_data = data[data['Cluster'] == i]
    plt.scatter(cluster_data['BestSquatKg'], cluster_data['BestDeadliftKg'], c=colors[i], label=f'Cluster {i}')

# Centroides de los clústeres
centroids = kmeans.cluster_centers_
centroids = scaler.inverse_transform(centroids)
plt.scatter(centroids[:, 0], centroids[:, 1], c='yellow', marker='X', s=200, label='Centroides')

plt.xlabel('Best Squat Kg')
plt.ylabel('Best Deadlift Kg')
plt.legend()
plt.title('Clustering usando K-means con k=2')
plt.show()
