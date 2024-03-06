import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

#Leemos el archivo de los datos
powerLift = pd.read_csv("powerLiftDataClean.csv")

#Obtenemos la media de cada columna BestSquatKg, BestDeadliftKg
mediaSquat = powerLift['BestSquatKg'].mean()
mediaDeadLift = powerLift['BestDeadliftKg'].mean()
#Guardamos los datos en un arreglo

datosMedias = [mediaSquat,mediaDeadLift]
listadoEjercicios = ["Squat","Dead Lift"]

#grafica de barras para ver media de ejercicios de levantamiento de pesas
fig, ax = plt.subplots()
ax.set_ylabel('Cantidad')
ax.set_title('Medias de cada levantamiento')

plt.bar(listadoEjercicios,datosMedias)
plt.savefig('graficaDeBarras.png')
plt.show()


