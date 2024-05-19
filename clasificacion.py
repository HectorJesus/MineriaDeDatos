import pandas as pd
from tabulate import tabulate
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import mode


def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))


def scatter_group_by(
    file_path: str, df: pd.DataFrame, x_column: str, y_column: str, label_column: str
):
    fig, ax = plt.subplots()
    labels = pd.unique(df[label_column])
    cmap = plt.get_cmap("hsv", len(labels) + 1)
    for i, label in enumerate(labels):
        filter_df = df.query(f"{label_column} == '{label}'")
        ax.scatter(filter_df[x_column], filter_df[y_column], label=label, color=cmap(i))
    ax.legend()
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    plt.legend()
    plt.title('Sentadilla / Peso muerto')
    plt.show()


def euclidean_distance(p_1: np.array, p_2: np.array) -> float:
    return np.sqrt(np.sum((p_2 - p_1) ** 2))


def k_nearest_neighbors(
    points: np.array, labels: np.array, input_data: np.array, k: int
):
    input_distances = np.array([[euclidean_distance(input_point, point) for point in points] for input_point in input_data])
    points_k_nearest = np.argsort(input_distances, axis=1)[:, :k]
    nearest_labels = np.array([[labels[idx] for idx in nearest] for nearest in points_k_nearest])
    return [mode(labels).mode[0] for labels in nearest_labels]


# Cargar el archivo CSV
df = pd.read_csv("powerLiftDataClean.csv")

# Crear las etiquetas necesarias para los datos existentes
# Aquí usamos un umbral basado en la mediana de BestSquatKg para dividir en dos grupos
threshold = df['BestSquatKg'].median()
df['label'] = df['BestSquatKg'].apply(lambda x: "grupo1" if x <= threshold else "grupo2")

# Visualizar los datos agrupados
scatter_group_by("img/groups.png", df, "BestSquatKg", "BestDeadliftKg", "label")

# Preparar datos para el algoritmo k-NN
points = df[['BestSquatKg', 'BestDeadliftKg']].values
labels = df['label'].values

# Nuevos puntos a clasificar
new_points = np.array([[100, 150], [1, 1], [1, 300], [80, 40], [400, 400]])

# Aplicar el algoritmo k-NN
k = 5
knn_labels = k_nearest_neighbors(points, labels, new_points, k)

# Mostrar los resultados
print(f"New points: {new_points}, Predicted labels: {knn_labels}")

# Visualización de los nuevos puntos clasificados
plt.scatter(df["BestSquatKg"], df["BestDeadliftKg"], c=df["label"].apply(lambda x: {'grupo1': 'red', 'grupo2': 'blue'}[x]), label='Existing data')
for i, point in enumerate(new_points):
    plt.scatter(point[0], point[1], c='yellow', edgecolors='k', label=f'New point {i+1} - {knn_labels[i]}')
plt.xlabel('BestSquatKg')
plt.ylabel('BestDeadliftKg')
plt.legend()
plt.title('k-NN Classification of New Points')
plt.show()
