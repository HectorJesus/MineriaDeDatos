import pandas as pd
import os
#Importamos la DB
powerLift = pd.read_csv("X_train.csv")

#Archivo final
csvpowerLift = 'powerLiftDataClean.csv'

#Comprobamos tipo de datos de la dataset
print(powerLift.dtypes)
#Convertimos la columna ['BestSquatKg'] a flotante
powerLift['BestSquatKg'] = pd.to_numeric(powerLift['BestSquatKg'], errors='coerce')

#Filtramos las columnas con las que trabajaremos
datosImportantes = powerLift[['BestSquatKg','BestDeadliftKg']]

#Verificamos la existencia del nuevo dataset
archivoExistencia = 'w' if not os.path.exists(csvpowerLift) else 'a'
#Guardamos las columnas con las que vamos a trabajar
datosImportantes.to_csv(csvpowerLift,index=False, mode=archivoExistencia)

#Comprobamos cambio de object a float de los datos finales
print(datosImportantes.dtypes)

