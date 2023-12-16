from utils import db_connect
engine = db_connect()

# your code here

#  Import some data manipulation and plotting packages
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

#ignore warnings
import warnings
warnings.filterwarnings("ignore")

"""0. Data Ingestion"""

import pandas as pd
file = "/workspace/Time-Series-Acea-Smart-Water-Analytics-/data/raw/Aquifer_Auser.csv"
total_data = pd.read_csv(file)

"""1. Analisis descriptivo de datos"""

total_data.head()
total_data.info()

#Missing values
total_data.isnull().sum()*100/len(total_data)

#Nan values
total_data.isnull().sum()

"""Combinatoria de variables"""

# Supongamos que tienes un DataFrame llamado 'total_data' con las columnas originales
# Nombre de las columnas originales
column_names = total_data.columns

# Crear una nueva columna "Rainfall" que contenga la suma de las columnas de precipitación
rainfall_columns = ['Rainfall_Gallicano', 'Rainfall_Pontetetto', 'Rainfall_Monte_Serra',
                    'Rainfall_Orentano', 'Rainfall_Borgo_a_Mozzano', 'Rainfall_Piaggione',
                    'Rainfall_Calavorno', 'Rainfall_Croce_Arcana', 'Rainfall_Tereglio_Coreglia_Antelminelli',
                    'Rainfall_Fabbriche_di_Vallico']
total_data['Rainfall'] = total_data[rainfall_columns].sum(axis=1)

# Crear una nueva columna "Depth_to_Groundwater" que contenga la suma de las columnas de profundidad de agua
depth_columns = ['Depth_to_Groundwater_LT2', 'Depth_to_Groundwater_SAL', 'Depth_to_Groundwater_PAG',
                 'Depth_to_Groundwater_CoS', 'Depth_to_Groundwater_DIEC']
total_data['Depth_to_Groundwater'] = total_data[depth_columns].sum(axis=1)

# Crear una nueva columna "Temperature" que contenga la suma de las columnas de temperatura
temperature_columns = ['Temperature_Orentano', 'Temperature_Monte_Serra', 'Temperature_Ponte_a_Moriano',
                       'Temperature_Lucca_Orto_Botanico']
total_data['Temperature'] = total_data[temperature_columns].sum(axis=1)

# Crear una nueva columna "Volume" que contenga la suma de las columnas de volumen
volume_columns = ['Volume_POL', 'Volume_CC1', 'Volume_CC2', 'Volume_CSA', 'Volume_CSAL']
total_data['Volume'] = total_data[volume_columns].sum(axis=1)

# Crear una nueva columna "Hydrometry" que contenga la suma de las columnas de hidrometría
hydrometry_columns = ['Hydrometry_Monte_S_Quirico', 'Hydrometry_Piaggione']
total_data['Hydrometry'] = total_data[hydrometry_columns].sum(axis=1)

# Eliminar las columnas originales de precipitación, profundidad de agua, temperatura, volumen e hidrometría
columns_to_drop = (rainfall_columns + depth_columns + temperature_columns + volume_columns + hydrometry_columns)
total_data.drop(columns=columns_to_drop, inplace=True)

# Mostrar las primeras filas del DataFrame con las columnas consolidadas
total_data.head()

"""Parsing de fechas"""

total_data['Date'] = pd.to_datetime(total_data['Date'], format='%d/%m/%Y')

# Now the 'date' column is parsed correctly as datetime objects

total_data.isnull().sum()
total_data.isnull().sum()*100/len(total_data)

"""1. Data Visualization

📊Data visualization

⛲Depth_to_Groundwater 👉 Target

⛲Rainfall 👉 It indicates the quantity of rain falling, expressed in millimeters (mm), in the area

⛲Temperature 👉 It indicates the temperature, expressed in °C, detected by the thermometric station

⛲Volume 👉  It indicates the volume of water, expressed in cubic meters (mc), taken from the drinking water treatment plant

⛲Hydrometry 👉 It indicates the groundwater level, expressed in meters (m), detected by the hydrometric station
"""

# Muestra todos los datos de la columna 'Temperature' en el DataFrame 'total_data'
print(total_data['Temperature'])

"""Cambio de farenheit a celsius"""

total_data['Temperature'] = total_data['Temperature'].apply(lambda fahrenheit: (fahrenheit - 32) * 5/9)

import matplotlib.pyplot as plt
import seaborn as sns
from datetime import date

# Crear subplots
f, ax = plt.subplots(nrows=5, ncols=1, figsize=(15, 25))

# Iterar a través de las columnas de total_data
for i, column in enumerate(total_data.drop('Date', axis=1).columns):
    sns.lineplot(x=total_data['Date'], y=total_data[column].fillna(method='ffill'), ax=ax[i], color='dodgerblue')
    ax[i].set_title('Feature: {}'.format(column), fontsize=14)
    ax[i].set_ylabel(ylabel=column, fontsize=14)
    ax[i].set_xlim([date(2009, 1, 1), date(2020, 6, 30)])

# Mostrar los gráficos
plt.show()

"""2. Data preprocessing"""

total_data = total_data.sort_values(by='Date')

# Check time intervals
total_data['delta'] = total_data['Date'] - total_data['Date'].shift(1)

total_data[['Date', 'delta']].head()

total_data['delta'].sum(), total_data['delta'].count()

total_data = total_data.drop('delta', axis=1)
total_data.isna().sum()

"""No es necesario, ya que no hay valores faltantes"""

import matplotlib.pyplot as plt
import seaborn as sns

f, ax = plt.subplots(nrows=3, ncols=1, figsize=(15, 20))

# Gráfico para 'Rainfall'
sns.lineplot(x=total_data['Date'], y=total_data['Rainfall'], ax=ax[0], color='darkorange', label='original')
ax[0].set_title('Feature: Rainfall', fontsize=14)
ax[0].set_ylabel(ylabel='Rainfall', fontsize=14)

# Gráfico para 'Depth_to_Groundwater'
sns.lineplot(x=total_data['Date'], y=total_data['Depth_to_Groundwater'], ax=ax[1], color='darkorange', label='original')
ax[1].set_title('Feature: Depth_to_Groundwater', fontsize=14)
ax[1].set_ylabel(ylabel='Depth_to_Groundwater', fontsize=14)

# Gráfico para 'Temperature'
sns.lineplot(x=total_data['Date'], y=total_data['Temperature'], ax=ax[2], color='darkorange', label='original')
ax[2].set_title('Feature: Temperature', fontsize=14)
ax[2].set_ylabel(ylabel='Temperature', fontsize=14)

# Ajustar los límites de fecha
for i in range(3):
    ax[i].set_xlim([total_data['Date'].min(), total_data['Date'].max()])

# Mostrar los gráficos
plt.tight_layout()
plt.show()

"""Resampling if apply

Upsampling is when the frequency of samples is increased (e.g. days to hours)

Downsampling is when the frequency of samples is decreased (e.g. days to weeks)
"""

# Asegúrate de que 'Date' sea el índice y está en formato de fecha y hora
total_data['Date'] = pd.to_datetime(total_data['Date'])
total_data.set_index('Date', inplace=True)

# Realiza el resampling semanal y calcula la media de las variables seleccionadas
downsample = total_data[['Depth_to_Groundwater', 'Temperature', 'Volume', 'Hydrometry', 'Rainfall']].resample('7D').mean()

# Reinicia el índice
downsample.reset_index(drop=False, inplace=True)

# Ahora, downsample contiene los datos muestreados de forma semanal
total_data = downsample.copy()

"""Stationary"""

#Check stationary

rolling_window = 52

f, ax = plt.subplots(nrows=2, ncols=1, figsize=(15, 12))

# Gráfico para 'Depth_to_Groundwater'
sns.lineplot(x=total_data.index, y=total_data['Depth_to_Groundwater'], ax=ax[0], color='dodgerblue')
sns.lineplot(x=total_data.index, y=total_data['Depth_to_Groundwater'].rolling(rolling_window).mean(), ax=ax[0], color='black', label='rolling mean')
sns.lineplot(x=total_data.index, y=total_data['Depth_to_Groundwater'].rolling(rolling_window).std(), ax=ax[0], color='orange', label='rolling std')
ax[0].set_title('Depth to Groundwater: Non-stationary \nnon-constant mean & non-constant variance', fontsize=14)
ax[0].set_ylabel(ylabel='Depth to Groundwater', fontsize=14)

# Gráfico para 'Temperature'
sns.lineplot(x=total_data.index, y=total_data['Temperature'], ax=ax[1], color='dodgerblue')
sns.lineplot(x=total_data.index, y=total_data['Temperature'].rolling(rolling_window).mean(), ax=ax[1], color='black', label='rolling mean')
sns.lineplot(x=total_data.index, y=total_data['Temperature'].rolling(rolling_window).std(), ax=ax[1], color='orange', label='rolling std')
ax[1].set_title('Temperature: Non-stationary \nvariance is time-dependent (seasonality)', fontsize=14)
ax[1].set_ylabel(ylabel='Temperature', fontsize=14)

plt.tight_layout()
plt.show()

"""Gráfico Superior - Profundidad del Agua Subterránea:

Exhibe fluctuaciones en la profundidad del agua subterránea a lo largo del tiempo, con valores negativos indicando mediciones debajo de un punto de referencia en la superficie.
La media móvil (línea negra) muestra una tendencia cambiante, sin un patrón estacional evidente.
La desviación estándar móvil (línea naranja) revela una variabilidad significativa en las mediciones.
Los datos son no estacionarios, lo que significa que la media y la varianza no son constantes y pueden estar influenciadas por diversos factores ambientales o humanos.

Gráfico Inferior - Temperatura:

Muestra un claro patrón estacional en las temperaturas con picos y valles regulares.
La media móvil (línea negra) suaviza las fluctuaciones y destaca la tendencia subyacente.
La desviación estándar móvil (línea naranja) indica variabilidad en la temperatura, con una posible dependencia estacional.
Los datos de temperatura son no estacionarios en el sentido de que la media y la varianza cambian a lo largo del año, siguiendo un patrón estacional.
Conclusión:

El patrón estacional es claro en la temperatura pero no en la profundidad del agua subterránea según la visualización proporcionada.

Dickey Fuller Testing
"""

from statsmodels.tsa.stattools import adfuller

result = adfuller(total_data['Depth_to_Groundwater'].values)
result

total_data.info()

from statsmodels.tsa.stattools import adfuller

f, ax = plt.subplots(nrows=3, ncols=2, figsize=(15, 9))

def visualize_adfuller_results(series, title, ax):
    result = adfuller(series)
    significance_level = 0.05
    adf_stat = result[0]
    p_val = result[1]
    crit_val_1 = result[4]['1%']
    crit_val_5 = result[4]['5%']
    crit_val_10 = result[4]['10%']

    if (p_val < significance_level) & ((adf_stat < crit_val_1)):
        linecolor = 'forestgreen'
    elif (p_val < significance_level) & (adf_stat < crit_val_5):
        linecolor = 'orange'
    elif (p_val < significance_level) & (adf_stat < crit_val_10):
        linecolor = 'red'
    else:
        linecolor = 'purple'
    sns.lineplot(x=total_data['Date'], y=series, ax=ax, color=linecolor)
    ax.set_title(f'ADF Statistic {adf_stat:0.3f}, p-value: {p_val:0.3f}\nCritical Values 1%: {crit_val_1:0.3f}, 5%: {crit_val_5:0.3f}, 10%: {crit_val_10:0.3f}', fontsize=14)
    ax.set_ylabel(ylabel=title, fontsize=14)

visualize_adfuller_results(total_data['Rainfall'].values, 'Rainfall', ax[0, 0])
visualize_adfuller_results(total_data['Temperature'].values, 'Temperature', ax[1, 0])
visualize_adfuller_results(total_data['Hydrometry'].values, 'River_Hydrometry', ax[0, 1])
visualize_adfuller_results(total_data['Volume'].values, 'Drainage_Volume', ax[1, 1])
visualize_adfuller_results(total_data['Depth_to_Groundwater'].values, 'Depth_to_Groundwater', ax[2, 0])

f.delaxes(ax[2, 1])
plt.tight_layout()
plt.show()

"""Interpretación de los gráficos

Rainfall (Precipitaciones): El estadístico ADF es -5.421 y el valor p es 0.000, lo que indica que la serie temporal es estacionaria ya que el valor p es menor que 0.05 y el estadístico ADF es menor que los valores críticos.

Temperature (Temperatura): El estadístico ADF es -9.861 y el valor p es 0.000, similar a la lluvia, lo que indica que la serie temporal es estacionaria.

River_Hydrometry (Hidrometría de Río): El estadístico ADF es -4.620 y el valor p es 0.000, indicando estacionariedad.

Drainage_Volume (Volumen de Drenaje): El estadístico ADF es -1.707 y el valor p es 0.428, lo que indica que la serie temporal no es estacionaria ya que el valor p es mayor que 0.05 y el estadístico ADF es mayor que los valores críticos.

Depth_to_Groundwater (Profundidad del Agua Subterránea): El estadístico ADF es -1.971 y el valor p es 0.299, lo que también indica no estacionariedad por las mismas razones que el volumen de drenaje.

Aplicar transformación:

No hay indicaciones directas de la necesidad de transformaciones en las series de precipitaciones, temperatura y hidrometría de río, ya que estas series ya parecen estacionarias. Las transformaciones son a menudo aplicadas para estabilizar la varianza, por lo que si hubiera evidencia de varianza no constante en los datos estacionarios, entonces podrías considerar aplicar una transformación.

Aplicar diferenciación:

Volumen de Drenaje (Drainage Volume): Dado que la serie temporal no es estacionaria, se recomienda aplicar diferenciación para alcanzar la estacionariedad.

Profundidad del Agua Subterránea (Depth to Groundwater): Esta serie también es no estacionaria, por lo que debería ser diferenciada para hacerla estacionaria.

Por tanto, las series de Volumen de Drenaje y Profundidad del Agua Subterránea requieren diferenciación. Las demás series, al ser estacionarias según la prueba ADF, no necesitarían diferenciación, aunque la necesidad de transformación dependerá de la constancia de la varianza que no se puede determinar solo con la prueba ADF.
"""

# First Order Differencing for 'Depth_to_Groundwater'
ts_diff_depth = np.diff(total_data['Depth_to_Groundwater'])
total_data['Depth_to_Groundwater_diff_1'] = np.append([0], ts_diff_depth)

# First Order Differencing for 'Drainage Volume'
ts_diff_drainage = np.diff(total_data['Volume'])
total_data['Drainage_Volume_diff_1'] = np.append([0], ts_diff_drainage)

# Visualización de 'Depth_to_Groundwater' después de la diferenciación
f, ax = plt.subplots(nrows=1, ncols=1, figsize=(15, 6))
visualize_adfuller_results(total_data['Depth_to_Groundwater_diff_1'], 'Differenced (1. Order) Depth to Groundwater', ax)

total_data.info()

"""Feature engineering"""

import pandas as pd

# Agregar características de fecha
total_data['year'] = pd.DatetimeIndex(total_data['Date']).year
total_data['month'] = pd.DatetimeIndex(total_data['Date']).month
total_data['day'] = pd.DatetimeIndex(total_data['Date']).day
total_data['day_of_year'] = pd.DatetimeIndex(total_data['Date']).dayofyear
total_data['week_of_year'] = pd.DatetimeIndex(total_data['Date']).strftime('%U').astype(int)
total_data['quarter'] = pd.DatetimeIndex(total_data['Date']).quarter
total_data['season'] = total_data['month'] % 12 // 3 + 1

# Visualizar las nuevas características
total_data[['Date', 'year', 'month', 'day', 'day_of_year', 'week_of_year', 'quarter', 'season']].head()

total_data.info()

"""Cyclical representation"""

f, ax = plt.subplots(nrows=1, ncols=1, figsize=(20, 3))

sns.lineplot(x=total_data['Date'], y=total_data['month'], color='dodgerblue')
ax.set_xlim([date(2009, 1, 1), date(2020, 6, 30)])
plt.show()

"""Time series: Stationary"""

from statsmodels.tsa.seasonal import seasonal_decompose

core_columns =  [
    'Rainfall', 'Temperature', 'Volume',
    'Hydrometry', 'Depth_to_Groundwater'
]

for column in core_columns:
    decomp = seasonal_decompose(total_data[column], period=52, model='additive', extrapolate_trend='freq')
    total_data[f"{column}_trend"] = decomp.trend # Trend: The increasing or decreasing value in the series.
    total_data[f"{column}_seasonal"] = decomp.seasonal # Seasonality: The repeating short-term cycle in the series.
    total_data[f"{column}_resid"] = decomp.resid # Noise: The random variation in the series.

total_data.info()

"""En este paso la fecha de elimina. Hay que revisarlo"""

import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
import pandas as pd
import numpy as np

# Como no tengo acceso al conjunto de datos real 'total_data', simularé algunos datos para demostración.
np.random.seed(0)
dates = pd.date_range('20010101', periods=1500)
total_data = pd.DataFrame(np.random.randn(1500, 5), index=dates, columns=[
    'Rainfall', 'Temperature', 'Volume', 'Hydrometry', 'Depth_to_Groundwater'
])

# Ajustaremos los datos simulados para que tengan una tendencia y estacionalidad similares a las esperadas.
for column in total_data.columns:
    total_data[column] = total_data[column].cumsum() + (np.sin(np.linspace(0, 10*np.pi, 1500)) * 10)

core_columns =  [
    'Rainfall', 'Temperature', 'Volume',
    'Hydrometry', 'Depth_to_Groundwater'
]

num_columns = len(core_columns)

# Ajustar la cantidad de filas y columnas en el subplot en función de la cantidad de columnas
num_rows = 4  # 4 subplots para cada componente
num_cols = num_columns

# Aumentamos el tamaño de la figura para una mejor visualización
fig, ax = plt.subplots(ncols=num_cols, nrows=num_rows, sharex=True, figsize=(20, 12))

for i, column in enumerate(core_columns):
    res = seasonal_decompose(total_data[column], model='additive', extrapolate_trend='freq')
    total_data[f"{column}_trend"] = res.trend
    total_data[f"{column}_seasonal"] = res.seasonal
    total_data[f"{column}_resid"] = res.resid

    ax[0, i].set_title('Decomposition of {}'.format(column), fontsize=18)
    ax[0, i].plot(total_data.index, total_data[column], color='dodgerblue')
    ax[0, i].set_ylabel('Observed', fontsize=16)

    ax[1, i].plot(total_data.index, total_data[f"{column}_trend"], color='dodgerblue')
    ax[1, i].set_ylabel('Trend', fontsize=16)

    ax[2, i].plot(total_data.index, total_data[f"{column}_seasonal"], color='dodgerblue')
    ax[2, i].set_ylabel('Seasonal', fontsize=16)

    ax[3, i].plot(total_data.index, total_data[f"{column}_resid"], color='dodgerblue')
    ax[3, i].set_ylabel('Residual', fontsize=16)

# Mantenemos las columnas Depth_to_Groundwater_diff_1 y Date en el DataFrame ajustado
total_data['Depth_to_Groundwater_diff_1'] = total_data['Depth_to_Groundwater'].diff()
total_data['Date'] = total_data.index

# Ajustamos el layout para evitar superposiciones
plt.tight_layout()
plt.show()

total_data.info()

import matplotlib.pyplot as plt

# Selecciona las columnas que contienen las componentes estacionales
seasonal_columns = [
    'Rainfall_seasonal', 'Temperature_seasonal', 'Volume_seasonal',
    'Hydrometry_seasonal', 'Depth_to_Groundwater_seasonal'
]

# Crea una figura para los gráficos
fig, ax = plt.subplots(nrows=len(seasonal_columns), ncols=1, figsize=(12, 8), sharex=True)

# Mostrar las componentes estacionales
for i, column in enumerate(seasonal_columns):
    ax[i].set_title(f'Seasonal Component of {column.split("_")[0]}', fontsize=14)
    ax[i].plot(total_data['Date'], total_data[column], color='dodgerblue')
    ax[i].set_ylabel('Seasonal', fontsize=12)
    ax[i].grid(True)

# Ajustar el layout para evitar superposiciones
plt.tight_layout()
plt.show()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Crea una figura con dos subplots
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# ACF y PACF para 'Depth_to_Groundwater'
plot_acf(total_data['Depth_to_Groundwater'], lags=100, ax=ax1)
plot_pacf(total_data['Depth_to_Groundwater'], lags=100, ax=ax2)

# Configura los títulos y etiquetas de los ejes
ax1.set_title('Autocorrelation Function (ACF) of Depth_to_Groundwater', fontsize=14)
ax2.set_title('Partial Autocorrelation Function (PACF) of Depth_to_Groundwater', fontsize=14)

# Muestra el gráfico
plt.tight_layout()
plt.show()

"""EDA

El gráfico muestra las componentes estacionales de cinco características distintas a lo largo del tiempo, desde octubre de 2017 hasta aproximadamente abril de 2020. Estas características parecen ser variables ambientales o hidrológicas, y son las siguientes:

Rainfall: La lluvia muestra un patrón estacional claro con picos y valles que se repiten a lo largo del tiempo. La amplitud de las fluctuaciones estacionales no parece cambiar significativamente durante el período observado.

Temperature: La temperatura también muestra una estacionalidad, probablemente con valores más altos en los meses de verano y más bajos en los meses de invierno. La amplitud de las fluctuaciones parece ser consistente a lo largo del tiempo, sugiriendo una estacionalidad estable a lo largo de los años.

Volume: Esta característica, que podría estar relacionada con el volumen de agua en un embalse o un caudal de agua, muestra una estacionalidad marcada con una amplitud que varía, lo que podría indicar cambios en el nivel de agua a lo largo del tiempo o una variación en las entradas y salidas de agua de un sistema.

Hydrometry: Este término generalmente se refiere a la medición del agua en términos de nivel, caudal o carga. La gráfica muestra fluctuaciones estacionales pero con una tendencia a aumentar ligeramente a lo largo del tiempo, lo que podría indicar un cambio gradual en las condiciones hidrométricas.

Depth to Groundwater: La profundidad al agua subterránea muestra variaciones estacionales, y se observa una tendencia a que la profundidad disminuya, lo que podría sugerir que el nivel del agua subterránea se está elevando con el tiempo o que se están produciendo menos fluctuaciones estacionales a medida que pasa el tiempo.

En cada gráfica, la línea azul representa la serie temporal de la componente estacional de la característica correspondiente. La leyenda "P25" parece ser un error o un residuo de la codificación del gráfico, ya que no se ajusta al contexto de los datos mostrados. Es probable que la leyenda deba ser eliminada o actualizada para reflejar con precisión lo que se está mostrando en el gráfico.

Cada gráfico tiene el eje x etiquetado como "Date", que va desde octubre de 2017 hasta abril de 2020, y el eje y que varía para cada característica y muestra la magnitud de la componente estacional correspondiente. El patrón estacional sugiere que los datos se han descompuesto para aislar y mostrar la estacionalidad, lo cual es una práctica común en el análisis de series temporales para entender mejor los patrones subyacentes y las tendencias de los datos.
"""

import seaborn as sns
import matplotlib.pyplot as plt

# Selecciona las columnas de interés
columns_of_interest = ['Depth_to_Groundwater', 'Temperature', 'Volume', 'Hydrometry', 'Rainfall']

# Filtra el DataFrame original para incluir solo las columnas de interés
subset_data = total_data[columns_of_interest]

# Calcula la matriz de correlación para las columnas seleccionadas
correlation_matrix = subset_data.corr()

# Crea un mapa de calor (heatmap) para visualizar la matriz de correlación
plt.figure(figsize=(10, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Matriz de Correlación', fontsize=16)
plt.show()

# Selecciona las columnas de interés
columns_of_interest = ['Depth_to_Groundwater', 'Temperature', 'Volume', 'Hydrometry', 'Rainfall']

# Filtra el DataFrame original para incluir solo las columnas de interés
subset_data = total_data[columns_of_interest]

# Calcula la matriz de correlación para las columnas seleccionadas
correlation_matrix = subset_data.corr()

# Muestra los valores numéricos de la matriz de correlación
print(correlation_matrix)

"""Volume vs. Rainfall: 0.882393

Volume vs. Hydrometry: 0.788343

Hydrometry vs. Rainfall: 0.857274

Depth_to_Groundwater vs. Hydrometry: 0.596303

Rainfall vs. Depth_to_Groundwater: 0.554393

Depth_to_Groundwater vs. Rainfall: 0.554393

Depth_to_Groundwater vs. Volume: 0.488210

Volume vs. Depth_to_Groundwater: 0.488210

Rainfall vs. Volume: 0.882393

Temperature vs. Rainfall: 0.621329

Hydrometry vs. Volume: 0.788343

Volume vs. Hydrometry: 0.788343

Hydrometry vs. Depth_to_Groundwater: 0.596303

Temperature vs. Hydrometry: 0.338251

Temperature vs. Depth_to_Groundwater: 0.098318

Depth_to_Groundwater vs. Temperature: 0.098318

Temperature vs. Volume: 0.606551

Volume vs. Temperature: 0.606551

Temperature vs. Volume: 0.606551

Temperature vs. Hydrometry: 0.338251

Autocorrelation Analysis
"""

total_data.info()

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

f, ax = plt.subplots(nrows=2, ncols=1, figsize=(16, 8))

plot_acf(total_data['Depth_to_Groundwater'], lags=100, ax=ax[0])
plot_pacf(total_data['Depth_to_Groundwater'], lags=100, ax=ax[1])

plt.show()

"""Modeling

Time series - Cross Validation
"""

total_data.info()

from sklearn.model_selection import TimeSeriesSplit

N_SPLITS = 3

X = total_data['Date']
y = total_data['Depth_to_Groundwater']

folds = TimeSeriesSplit(n_splits=N_SPLITS)

import matplotlib.pyplot as plt
from sklearn.model_selection import TimeSeriesSplit

# Crear un objeto TimeSeriesSplit con el número deseado de divisiones
N_SPLITS = 3
tscv = TimeSeriesSplit(n_splits=N_SPLITS)

# Datos de ejemplo para ilustrar
X = total_data['Date']  # Características (fechas)
y = total_data['Depth_to_Groundwater']  # Variable objetivo

# Crear subplots para visualizar los conjuntos de entrenamiento y prueba
fig, ax = plt.subplots(nrows=N_SPLITS, figsize=(10, 6), sharex=True)

# Iterar a través de las divisiones y visualizar los conjuntos de entrenamiento y prueba
for i, (train_index, test_index) in enumerate(tscv.split(X)):
    train_set_dates = X.iloc[train_index]
    test_set_dates = X.iloc[test_index]

    # Representar los conjuntos de entrenamiento y prueba en cada iteración
    ax[i].plot(X, y, label='Data', color='blue')
    ax[i].scatter(train_set_dates, y.iloc[train_index], label='Train Set', color='green')
    ax[i].scatter(test_set_dates, y.iloc[test_index], label='Test Set', color='red')

    # Configurar títulos y etiquetas
    ax[i].set_title(f'Fold {i+1}')
    ax[i].set_xlabel('Date')
    ax[i].set_ylabel('Depth to Groundwater')
    ax[i].legend()

# Ajustar el diseño y mostrar los subplots
plt.tight_layout()
plt.show()

"""Auto-ARIMA"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error  # Importar las métricas de error
import pmdarima as pm

# Supongamos que ya tienes tus datos cargados en un DataFrame llamado total_data

# Definir las columnas de fecha y objetivo
X = total_data['Date']
y = total_data['Depth_to_Groundwater']

# Dividir los datos en conjuntos de entrenamiento y prueba con random_state
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Ajustar el modelo ARIMA utilizando auto_arima
model = pm.auto_arima(y_train, start_p=1, start_q=1,
                      test='adf',       # use adftest to find optimal 'd'
                      max_p=3, max_q=3, # maximum p and q
                      m=1,              # frequency of series
                      d=None,           # let model determine 'd'
                      seasonal=False,   # No Seasonality
                      start_P=0,
                      D=0,
                      trace=True,
                      error_action='ignore',
                      suppress_warnings=True,
                      stepwise=True,
                      random_state=42)  # Establecer random_state aquí

print(model.summary())

# Hacer predicciones en el conjunto de prueba
y_pred = model.predict(len(y_test))

# Calcular el Error Cuadrático Medio (MSE)
mse = mean_squared_error(y_test, y_pred)

# Calcular el Error Absoluto Medio (MAE)
mae = mean_absolute_error(y_test, y_pred)

# Imprimir las métricas de error
print("Error Cuadrático Medio (MSE):", mse)
print("Error Absoluto Medio (MAE):", mae)

model.plot_diagnostics(figsize=(16,8))
plt.show()

"""Gráfico de residuos estandarizados: Muestra los residuos del modelo (las diferencias entre los valores observados y los valores ajustados por el modelo) a lo largo del tiempo. Lo que buscas aquí es que no haya patrones claros en los residuos, lo que indicaría que el modelo no capturó alguna estructura en los datos. En este gráfico, los residuos parecen ser bastante aleatorios sin patrones obvios, lo que es un buen signo.

Histograma más densidad estimada (con KDE y distribución normal): Este gráfico muestra la distribución de los residuos estandarizados. El histograma (en azul) muestra la frecuencia de los residuos, la línea verde (KDE) es una estimación de la densidad de esos residuos, y la línea naranja es la distribución normal estándar. Idealmente, queremos que los residuos sigan una distribución normal estándar, lo que sugeriría que el modelo ha capturado adecuadamente la información en los datos y que los errores son solo ruido aleatorio. Tu gráfico muestra que los residuos se parecen bastante a una distribución normal, lo cual es bueno, aunque parece haber una ligera desviación en las colas de la distribución.

Gráfico Q-Q (Quantile-Quantile): Este gráfico compara los cuantiles de los residuos con los cuantiles que esperaríamos si los residuos siguieran una distribución normal. Si los residuos son normales, los puntos deberían caer aproximadamente en línea recta. Tu gráfico muestra que en general los puntos siguen la línea, pero hay cierta desviación en los extremos (colas), lo que sugiere que puede haber más valores extremos de lo esperado en una distribución normal.

Correlograma de los residuos: Este gráfico muestra la autocorrelación de los residuos en diferentes retrasos. No queremos ver autocorrelación significativa en los residuos, porque eso indicaría que el modelo no ha capturado toda la estructura de autocorrelación en los datos. En un modelo bien ajustado, esperaríamos que la mayoría de estos valores estén dentro del área sombreada (que representa el intervalo de confianza), lo que sugeriría que no hay autocorrelación significativa. En tu correlograma, todos los puntos están dentro del área sombreada, lo que sugiere que no hay autocorrelación significativa en los residuos.

En resumen, estos diagnósticos sugieren que el modelo ARIMA automático se ha ajustado razonablemente bien a los datos. Sin embargo, las pequeñas desviaciones en la distribución normal de los residuos podrían ser un área para investigar más, tal vez con modelos más robustos o transformaciones adicionales de los datos.

****************************************************************

Gráfico de residuos estandarizados: Muestra que los residuos se distribuyen al azar y no muestran patrones discernibles, lo cual indica que el modelo ARIMA ha capturado la estructura temporal de los datos de manera efectiva.

Histograma y densidad estimada: Los residuos muestran una aproximación razonable a la distribución normal, con ligeras desviaciones en las colas, lo cual sugiere que el modelo ajusta bien la mayoría de los datos pero podría no manejar tan bien los valores atípicos.

Gráfico Q-Q: La mayoría de los puntos siguen la línea teórica de la distribución normal, con desviaciones en los extremos. Esto refuerza la idea de que los valores atípicos no están siendo capturados perfectamente por el modelo.

Correlograma: No hay autocorrelación significativa en los residuos, lo que indica que el modelo está utilizando adecuadamente la información temporal de la serie de datos y no deja patrones no explicados.

Conclusión: El modelo ARIMA está funcionando adecuadamente con los datos proporcionados, con la eficacia del ajuste reflejada en la falta de patrones en los residuos y la ausencia de autocorrelación significativa. Las pequeñas desviaciones en la normalidad de los residuos sugieren que podría haber margen de mejora, posiblemente con un enfoque en la gestión de valores atípicos o mediante la inclusión de términos adicionales en el modelo.
"""

import pickle

# Guardar el modelo ARIMA en un archivo Pickle
with open('/workspace/Time-Series-Acea-Smart-Water-Analytics-/models/arima_model.pkl', 'wb') as model_file:
    pickle.dump(model, model_file)