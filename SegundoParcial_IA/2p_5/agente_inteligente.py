#pip install pandas sqlalchemy scikit-learn tensorflow

from sqlalchemy import create_engine
import pandas as pd

# Reemplaza con tu cadena de conexión
DATABASE_URI = 'sqlite:///world.sql'
engine = create_engine(DATABASE_URI)

# Cargar los datos en un DataFrame de pandas
query = "SELECT * FROM city"
df = pd.read_sql(query, engine)

#Paso 2: Conexión a la Base de Datos-----------------------------------------------------------------------------------
from sqlalchemy import create_engine
import pandas as pd

# Reemplaza con tu cadena de conexión
DATABASE_URI = 'sqlite:///your_database.db'
engine = create_engine(DATABASE_URI)

# Cargar los datos en un DataFrame de pandas
query = "SELECT * FROM your_table"
df = pd.read_sql(query, engine)

#Paso 3: Preprocesamiento de Datos----------------------------------------------------------------------------
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Supongamos que 'category' y 'brand' son variables categóricas
df['category'] = LabelEncoder().fit_transform(df['category'])
df['brand'] = LabelEncoder().fit_transform(df['brand'])

# Seleccionamos las características y las etiquetas
features = df.drop(columns=['price'])  # 'price' es la etiqueta a predecir
labels = df['price']

# Normalizamos las características
scaler = StandardScaler()
features = scaler.fit_transform(features)

#Paso 4: Construcción y Entrenamiento de la Red Neuronal------------------------------------------------
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Construcción del modelo
model = Sequential([
    Dense(64, input_shape=(features.shape[1],), activation='relu'),
    Dense(64, activation='relu'),
    Dense(1)
])

model.compile(optimizer='adam', loss='mse')

# Entrenamiento del modelo
model.fit(features, labels, epochs=10, batch_size=32)

#Paso 5: Búsqueda y Comparación con la Red Neuronal-------------------------------------------------------
def search_and_compare_nn(df, search_params, compare_fields, model, scaler):
    """
    Realiza una búsqueda en el DataFrame y compara los resultados usando una red neuronal.

    :param df: DataFrame con los datos.
    :param search_params: Diccionario con los parámetros de búsqueda.
    :param compare_fields: Lista de campos a comparar.
    :param model: Modelo de red neuronal entrenado.
    :param scaler: Objeto StandardScaler usado para normalizar los datos.
    :return: DataFrame con los resultados comparados.
    """
    # Filtrar el DataFrame según los parámetros de búsqueda
    filtered_df = df
    for key, value in search_params.items():
        filtered_df = filtered_df[filtered_df[key] == value]
    
    # Seleccionar los campos a comparar
    comparison_df = filtered_df[compare_fields]
    
    # Normalizar los datos de comparación
    comparison_features = scaler.transform(comparison_df)
    
    # Predecir los precios usando el modelo
    predicted_prices = model.predict(comparison_features)
    
    # Añadir las predicciones al DataFrame
    comparison_df['predicted_price'] = predicted_prices
    
    return comparison_df

# Parámetros de búsqueda y campos a comparar
search_params = {'category': 0, 'brand': 1}  # Ajusta estos valores según tu codificación
compare_fields = ['feature1', 'feature2']  # Reemplaza con los nombres reales de tus características

# Realizar la búsqueda y comparación
results = search_and_compare_nn(df, search_params, compare_fields, model, scaler)
print(results)

#Paso 6: Presentación de los Resultados---------------------------------------------------------------------
from tabulate import tabulate

def print_results(results):
    print(tabulate(results, headers='keys', tablefmt='grid'))

# Imprimir los resultados de la búsqueda
print_results(results)
