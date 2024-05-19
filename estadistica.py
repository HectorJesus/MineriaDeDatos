#La estadistica descriptiva e utiliza para 
# resumir y describir las características básicas 
# de un conjunto de datos. 

import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('powerLiftDataClean.csv')

#Mostramos en consola la primera fila de la dataset
print(data.head())

#Ahora mostraremos un resumen estadistico
rEst = data.describe()
print(rEst)

#Medias de tendencia central y descripcion
# Media
mean = data.mean()
print('Media:', mean)

# Mediana
median = data.median()
print('Mediana:', median)

# Moda
mode = data.mode()
print('Moda:', mode)

# Varianza
variance = data.var()
print('Varianza:', variance)

# Desviación estándar
std_dev = data.std()
print('Desviación Estándar:', std_dev)

# Rango
data_range = data.max() - data.min()
print('Rango:', data_range)

#Creamos graficas para la visualizacion de datos

# Histograma
data.hist(figsize=(10, 8))
plt.suptitle('Histograma de Variables')
plt.show()

# Gráfico de cajas 
data.boxplot(figsize=(10, 8))
plt.title('Gráfico de Cajas')
plt.show()
