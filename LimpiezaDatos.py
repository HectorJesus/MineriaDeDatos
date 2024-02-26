import pandas as pd
#Importamos la DB y la imprimimos
powerLift = pd.read_csv("X_train.csv")

print(powerLift)

#Filtramos el genero de la db
dbm = powerLift["Sex"]

print(dbm)

#Obtenemos el numero de participantes segun el genero

dbc = powerLift["Sex"].value_counts()

print(dbc)