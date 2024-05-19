# Forecasting - el proceso de usar modelos matemáticos y estadísticas 
# para predecir futuros valores basados en datos históricos. 
# MODELO ARIMA


import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error

#Leemos el archivo de los datos
powerLift = pd.read_csv("powerLiftDataClean.csv", parse_dates=['BestSquatKg'],index_col=['BestSquatKg'])
plt.figure(figsize=(12,6))
plt.plot(powerLift, label='Constacia En Pesos')
plt.xlabel('BestSquatKg')
plt.ylabel('BestDeadliftKg')
plt.title('Constancia En Pesos')
plt.legend()
plt.show()

divData = int(len(powerLift)*0.8)
train, test = powerLift[0:divData], powerLift[divData:]

model = ARIMA(train, order=(5,1,0))
model_fit = model.fit()

forecast = model_fit.forecast(steps=len(test))
forecast = pd.Series(forecast, index=test.index)

error = mean_squared_error(test, forecast)
print('Test MSE: %.3f' % error)

plt.figure(figsize=(12, 6))
plt.plot(train, label='Entrenamiento')
plt.plot(test, label='Prueba')
plt.plot(forecast, label='Pronóstico', color='red')
plt.xlabel('Fecha')
plt.ylabel('Ventas')
plt.title('Pronóstico de Ventas Diarias usando ARIMA')
plt.legend()
plt.show()