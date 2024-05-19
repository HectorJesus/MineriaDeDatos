import pandas as pd
import os
#Importamos la DB
powerLift = pd.read_csv("X_train.csv")
df = pd.DataFrame(powerLift)
#Archivo final
csvpowerLift = 'DataText.csv'

#Comprobamos tipo de datos de la dataset
print(df.dtypes)

df['Name'] = df['Name'].astype(str)
df['Sex'] = df['Sex'].astype(str)

#Filtramos las columnas con las que trabajaremos
datosImportantes = df[['Name','Sex']]


#Verificamos la existencia del nuevo dataset
archivoExistencia = 'w' if not os.path.exists(csvpowerLift) else 'a'
#Guardamos las columnas con las que vamos a trabajar
datosImportantes.to_csv(csvpowerLift,index=False, mode=archivoExistencia)

# Contar la cantidad de veces que se repite cada texto en la columna
conteo = datosImportantes['Sex'].value_counts()
print("Conteo de hombres y Mujeres en la competencia:")
print(conteo)
