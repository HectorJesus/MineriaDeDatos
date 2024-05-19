import matplotlib.pyplot as plt
import statsmodels.api as sm
import numbers
import pandas as pd
from tabulate import tabulate
import os

def print_tabulate(df: pd.DataFrame):
    print(tabulate(df, headers=df.columns, tablefmt="orgtbl"))

def transform_variable(df: pd.DataFrame, x:str)->pd.Series:
    if isinstance(df[x][0], numbers.Number):
        return df[x] # type: pd.Series
    else:
        return pd.Series([i for i in range(0, len(df[x]))])


def linear_regression(df: pd.DataFrame, x:str, y: str)->None:
    fixed_x = transform_variable(df, x)
    model= sm.OLS(df[y],sm.add_constant(fixed_x)).fit()
    print(model.summary())

    coef = pd.read_html(model.summary().tables[1].as_html(),header=0,index_col=0)[0]['coef']
    df.plot(x=x,y=y, kind='scatter')
    plt.plot(df[x],[pd.DataFrame.mean(df[y]) for _ in fixed_x.items()], color='green')
    plt.plot(df_by_sal[x],[ coef.values[1] * x + coef.values[0] for _, x in fixed_x.items()], color='red')
    plt.xticks(rotation=90)
    plt.savefig(f'img/lr_{y}_{x}.png')
    plt.close()

#Importamos la DB normal
powerLift = pd.read_csv("X_train.csv")

#Archivo final
csvpowerLift = 'DataSquad.csv'

#Comprobamos tipo de datos de la dataset
print(powerLift.dtypes)
#Convertimos la columna ['Age'] a flotante
powerLift['Age'] = pd.to_numeric(powerLift['Age'], errors='coerce')
#Convertimos la columna ['BestSquatKg'] a flotante
powerLift['BestSquatKg'] = pd.to_numeric(powerLift['BestSquatKg'], errors='coerce')

#Filtramos las columnas con las que trabajaremos
datosImportantes = powerLift[['Age','BestSquatKg']]

#Verificamos la existencia del nuevo dataset
archivoExistencia = 'w' if not os.path.exists(csvpowerLift) else 'a'
#Guardamos las columnas con las que vamos a trabajar
datosImportantes.to_csv(csvpowerLift,index=False, mode=archivoExistencia)

#Comprobamos cambio de object a float de los datos finales
print(datosImportantes.dtypes)


df = pd.read_csv("DataSquad.csv") 
df_by_sal = df.groupby("Age")\
              .aggregate(BestSquatKg=pd.NamedAgg(column="BestSquatKg", aggfunc=pd.DataFrame.max))
df_by_sal["BestSquatKg"] = df_by_sal["BestSquatKg"]**10
df_by_sal.reset_index(inplace=True)
print_tabulate(df_by_sal.head())
linear_regression(df_by_sal, "Age", "BestSquatKg")
