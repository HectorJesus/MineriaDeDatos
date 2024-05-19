
#Prueba t de Student: Se utiliza para comparar las medias 
#de dos grupos independientes para determinar si hay una
#diferencia significativa entre ellas.

import pandas as pd
from scipy.stats import ttest_ind

# Cargar el dataset
data = pd.read_csv("powerLiftDataClean.csv")

# Eliminar filas con valores faltantes
data.dropna(subset=["BestSquatKg", "BestDeadliftKg"], inplace=True)

# Seleccionar las columnas para la prueba
squat_kg = data["BestSquatKg"]
deadlift_kg = data["BestDeadliftKg"]

# Verificar la varianza de las muestras
if squat_kg.var() == 0 or deadlift_kg.var() == 0:
    print("Una de las muestras tiene varianza cero, no se puede realizar la prueba.")
else:
    # Realizar la prueba t de Student
    t_statistic, p_value = ttest_ind(squat_kg, deadlift_kg)
    print("Estadística t:", t_statistic)
    print("Valor p:", p_value)

#En este caso al ser negativo la estadistica nos indica
# que la diferencua entre las medias de las dos muestras
#es negativa, osea que la primera muestra tiene vañpres
#mas bajos que la segunda muestra