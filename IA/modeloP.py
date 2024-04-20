#////////////////////////////////////////////////////////////////////////////////////////////////////
# MODELO INICIAL
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.datasets import make_classification
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Genera datos de ejemplo
# X, y = make_classification(n_samples=1000, n_features=20, random_state=42)

# # Divide los datos en conjunto de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Crea y entrena el modelo
# model = HistGradientBoostingClassifier()
# model.fit(X_train, y_train)

# # Realiza predicciones en el conjunto de prueba
# predictions = model.predict(X_test)

# # Evalúa la precisión del modelo
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy:", accuracy)

#////////////////////////////////////////////////////////////////////////////////////////////////////
#MODELO PARA CNESTAR CON SQL LITE
# import sqlite3
# import pandas as pd
# from sklearn.ensemble import HistGradientBoostingClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Cargar datos desde un archivo CSV (cambia 'datos.csv' por el nombre de tu archivo)
# data = pd.read_csv('datos.csv')

# # Dividir los datos en características (X) y etiquetas (y)
# X = data.drop(columns=['target'])
# y = data['target']

# # Dividir los datos en conjunto de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Crea y entrena el modelo
# model = HistGradientBoostingClassifier()
# model.fit(X_train, y_train)

# # Conectar a la base de datos (creará un archivo si no existe)
# conn = sqlite3.connect('datos.db')

# # Crear un cursor para ejecutar comandos SQL
# cursor = conn.cursor()

# # Crear una tabla para almacenar predicciones
# cursor.execute('''CREATE TABLE IF NOT EXISTS predicciones
#                   (id INTEGER PRIMARY KEY, entrada TEXT, prediccion INTEGER)''')

# # Insertar datos en la tabla y hacer predicciones
# for i, entrada in X_test.iterrows():
#     entrada_str = ",".join(map(str, entrada))
#     prediccion = model.predict([entrada])[0]
#     cursor.execute("INSERT INTO predicciones (entrada, prediccion) VALUES (?, ?)", (entrada_str, prediccion))

# # Guardar los cambios
# conn.commit()

# # Consultar todas las predicciones
# cursor.execute("SELECT * FROM predicciones")
# predicciones = cursor.fetchall()
# print("Predicciones en la base de datos:")
# for prediccion in predicciones:
#     print(prediccion)

# # Evaluar la precisión del modelo
# accuracy = accuracy_score(y_test, model.predict(X_test))
# print("Accuracy:", accuracy)

# # Cerrar la conexión
# conn.close()
#////////////////////////////////////////////////////////////////////////////////////////////////////
#MODELO FINAL 1.0
# import sqlite3
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score

# # Conectar a la base de datos
# conn = sqlite3.connect('tu_basededatos.db')

# # Cargar datos desde la tabla de revisiones
# data = pd.read_sql_query("SELECT * FROM revisiones", conn)

# # Preprocesamiento de datos
# # Seleccionar características relevantes
# features = ['review_text', 'rating', 'age']
# data = data[features + ['recommend_index']]  # Incluye solo las características relevantes

# # Convertir la calificación del producto y la edad del revisor a valores numéricos
# data['rating'] = pd.to_numeric(data['rating'])
# data['age'] = pd.to_numeric(data['age'], errors='coerce')  # Si hay valores no numéricos, se convierten en NaN

# # Eliminar filas con valores faltantes
# data = data.dropna()

# # Dividir los datos en conjunto de características (X) y etiquetas (y)
# X = data[['review_text', 'rating', 'age']]
# y = data['recommend_index']

# # Dividir los datos en conjunto de entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Vectorización de texto
# vectorizer = TfidfVectorizer(max_features=1000)
# X_train_text = vectorizer.fit_transform(X_train['review_text'])
# X_test_text = vectorizer.transform(X_test['review_text'])

# # Concatenar características numéricas con características de texto
# X_train_features = pd.concat([pd.DataFrame(X_train_text.toarray()), X_train[['rating', 'age']].reset_index(drop=True)], axis=1)
# X_test_features = pd.concat([pd.DataFrame(X_test_text.toarray()), X_test[['rating', 'age']].reset_index(drop=True)], axis=1)

# # Entrenamiento del modelo (Random Forest)
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train_features, y_train)

# # Predicciones en el conjunto de prueba
# predictions = model.predict(X_test_features)

# # Evaluación del modelo
# accuracy = accuracy_score(y_test, predictions)
# print("Accuracy:", accuracy)

# # Cerrar la conexión
# conn.close()
#////////////////////////////////////////////////////////////////////////////////////////////////////
#MODELO FINALCON BD DE VJ 1.0
# import numpy as np
# import pandas as pd
# import seaborn as sns
# import matplotlib.pyplot as plt
# from plotly.offline import init_notebook_mode, iplot
# import plotly.graph_objs as go
# from wordcloud import WordCloud
# import sqlite3


# # Cargar la base de datos desde un archivo CSV (especificando la ruta completa)
# df = pd.read_csv(r'C:\Users\alexi\Documents\Uni\vgsales.csv')

# # Preprocesamiento de datos
# df.dropna(inplace=True)  # Eliminar filas con valores nulos

# # Función para encontrar el juego más vendido en cada grupo
# def juego_mas_vendido_por_grupo(data, grupo):
#     mas_vendido = data.groupby(grupo)['Global_Sales'].idxmax()  # Índice del juego más vendido en cada grupo
#     return data.loc[mas_vendido]

# # Juego más vendido por plataforma
# juego_mas_vendido_plataforma = juego_mas_vendido_por_grupo(df, 'Platform')

# # Juego más vendido por género
# juego_mas_vendido_genero = juego_mas_vendido_por_grupo(df, 'Genre')

# # Juego más vendido por editor
# juego_mas_vendido_editor = juego_mas_vendido_por_grupo(df, 'Publisher')

# # Juego más vendido por año
# juego_mas_vendido_anio = juego_mas_vendido_por_grupo(df, 'Year')

# # Visualizar resultados (puedes imprimir o hacer otros tipos de visualizaciones)
# print("Juego más vendido por plataforma:")
# print(juego_mas_vendido_plataforma[['Platform', 'Name', 'Global_Sales']])
# print("\nJuego más vendido por género:")
# print(juego_mas_vendido_genero[['Genre', 'Name', 'Global_Sales']])
# print("\nJuego más vendido por editor:")
# print(juego_mas_vendido_editor[['Publisher', 'Name', 'Global_Sales']])
# print("\nJuego más vendido por año:")
# print(juego_mas_vendido_anio[['Year', 'Name', 'Global_Sales']])

# # Establecer conexión con la base de datos SQLite
# conn = sqlite3.connect('tu_base_de_datos.db')

# # Consulta SQL para seleccionar todos los datos de la tabla principal
# query = "SELECT * FROM Name"

# # Leer los datos de la base de datos utilizando la consulta SQL y la conexión
# data = pd.read_sql_query(query, conn)

# # Cerrar la conexión
# conn.close()

# # Mostrar los datos
# print(data)

#////////////////////////////////////////////////////////////////////////////////////////////////////
#MODELO FINALCON BD DE VJ 1.1

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Cargar datos históricos
# df = pd.read_csv(r'C:\Users\alexi\Documents\Uni\vgsales.csv')

# # Preprocesamiento de datos
# df.dropna(inplace=True)  # Eliminar filas con valores nulos

# # Características relevantes para la predicción
# features = ['Platform', 'Genre', 'Year']  # Puedes agregar más características si es necesario

# # División de datos en características (X) y etiquetas (y)
# X = df[features]
# y = df['Name']  # La etiqueta será el nombre del juego más vendido

# # División de datos en entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Codificación one-hot para las características categóricas
# X_train_encoded = pd.get_dummies(X_train)
# X_test_encoded = pd.get_dummies(X_test)

# # Entrenamiento del modelo de clasificación
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train_encoded, y_train)

# # Evaluación del modelo
# y_pred = model.predict(X_test_encoded)
# accuracy = accuracy_score(y_test, y_pred)
# print("Precisión del modelo:", accuracy)

# # Predicción para los años 2021 y 2022
# datos_prediccion = pd.DataFrame({'Platform': ['Plataforma_predicha_2021', 'Plataforma_predicha_2022'],
#                                  'Genre': ['Género_predicho_2021', 'Género_predicho_2022'],
#                                  'Year': [2021, 2022]})
# datos_prediccion_encoded = pd.get_dummies(datos_prediccion)
# prediccion_juegos = model.predict(datos_prediccion_encoded)
# print("Predicción de juegos más vendidos para 2021 y 2022:", prediccion_juegos)

#////////////////////////////////////////////////////////////////////////////////////////////////////
#MODELO FINALCON BD DE VJ y tomando datos random para poder procesarlos 1.2
#FUNCIONAL PERO POCA PRECISIÓN

# import numpy as np
# import pandas as pd
# from sklearn.model_selection import train_test_split
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.metrics import accuracy_score

# # Cargar datos históricos
# df = pd.read_csv(r'C:\Users\alexi\Documents\Uni\vgsales.csv')

# # Preprocesamiento de datos
# df.dropna(inplace=True)  # Eliminar filas con valores nulos

# # Características relevantes para la predicción
# features = ['Platform', 'Genre', 'Year']  # Puedes agregar más características si es necesario

# # Reducir el tamaño del conjunto de datos
# df_sample = df.sample(frac=0.5, random_state=42)  # Reducir al 50% del tamaño original

# # División de datos en características (X) y etiquetas (y)
# X = df_sample[features]
# y = df_sample['Name']  # La etiqueta será el nombre del juego más vendido

# # División de datos en entrenamiento y prueba
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# # Codificación one-hot para las características categóricas (datos de entrenamiento)
# X_train_encoded = pd.get_dummies(X_train)

# # Codificación one-hot para las características categóricas (datos de prueba)
# X_test_encoded = pd.get_dummies(X_test)

# # Asegurar que los datos de prueba tengan las mismas columnas que los datos de entrenamiento
# missing_cols_test = set(X_train_encoded.columns) - set(X_test_encoded.columns)
# for col in missing_cols_test:
#     X_test_encoded[col] = 0
# X_test_encoded = X_test_encoded[X_train_encoded.columns]

# # Entrenamiento del modelo de clasificación
# model = RandomForestClassifier(n_estimators=100, random_state=42)
# model.fit(X_train_encoded, y_train)

# # Evaluación del modelo
# y_pred = model.predict(X_test_encoded)
# accuracy = accuracy_score(y_test, y_pred)
# print("Precisión del modelo:", accuracy)

# # Predicción para los años 2021 y 2022
# datos_prediccion = pd.DataFrame({'Platform': ['Plataforma_predicha_2021', 'Plataforma_predicha_2022'],
#                                  'Genre': ['Género_predicho_2021', 'Género_predicho_2022'],
#                                  'Year': [2021, 2022]})

# # Codificación one-hot para las características categóricas (datos de predicción)
# datos_prediccion_encoded = pd.get_dummies(datos_prediccion)

# # Asegurar que los datos de predicción tengan las mismas columnas que los datos de entrenamiento
# missing_cols_prediccion = set(X_train_encoded.columns) - set(datos_prediccion_encoded.columns)
# for col in missing_cols_prediccion:
#     datos_prediccion_encoded[col] = 0
# datos_prediccion_encoded = datos_prediccion_encoded[X_train_encoded.columns]

# # Realizar la predicción de los juegos más vendidos para los años 2021 y 2022
# prediccion_juegos = model.predict(datos_prediccion_encoded)
# print("Predicción de juegos más vendidos para 2021 y 2022:", prediccion_juegos)

#////////////////////////////////////////////////////////////////////////////////////////////////////
#MODELO FINALCON BD DE VJ y tomando datos random para poder procesarlos 1.3

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import warnings

# Desactivar advertencias de sklearn
warnings.filterwarnings("ignore")

# Cargar datos históricos
df = pd.read_csv(r'vgsales.csv')

# Preprocesamiento de datos
df.dropna(inplace=True)  # Eliminar filas con valores nulos

# Características relevantes para la predicción
features = ['Platform', 'Genre', 'Year']  # Puedes agregar más características si es necesario

# Reducir el tamaño del conjunto de datos
df_sample = df.sample(frac=0.5, random_state=42)  # Reducir al 50% del tamaño original

# Eliminar clases poco representadas
df_sample = df_sample.groupby('Name').filter(lambda x: len(x) > 1)

# División de datos en características (X) y etiquetas (y)
X = df_sample[features]
y = df_sample['Name']  # La etiqueta será el nombre del juego más vendido

# División de datos en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Codificación one-hot para las características categóricas (datos de entrenamiento)
X_train_encoded = pd.get_dummies(X_train)

# Codificación one-hot para las características categóricas (datos de prueba)
X_test_encoded = pd.get_dummies(X_test)

# Asegurar que los datos de prueba tengan las mismas columnas que los datos de entrenamiento
missing_cols_test = set(X_train_encoded.columns) - set(X_test_encoded.columns)
for col in missing_cols_test:
    X_test_encoded[col] = 0
X_test_encoded = X_test_encoded[X_train_encoded.columns]

# Definir los parámetros que deseas ajustar
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5, 10]
}

# Inicializar el modelo RandomForestClassifier
rf = RandomForestClassifier(random_state=42)

# Realizar la búsqueda grid
grid_search = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
grid_search.fit(X_train_encoded, y_train)

# Imprimir los mejores parámetros encontrados
print("Mejores hiperparámetros encontrados:", grid_search.best_params_)

# Entrenar el modelo con los mejores hiperparámetros
best_rf = grid_search.best_estimator_
best_rf.fit(X_train_encoded, y_train)

# Evaluar el modelo
y_pred = best_rf.predict(X_test_encoded)
accuracy = accuracy_score(y_test, y_pred)
print("Precisión del modelo después de la búsqueda grid:", accuracy)

# Predicción para los años 2021 y 2022
datos_prediccion = pd.DataFrame({'Platform': ['Plataforma_predicha_2021', 'Plataforma_predicha_2022'],
                                 'Genre': ['Género_predicho_2021', 'Género_predicho_2022'],
                                 'Year': [2021, 2022]})

# Codificación one-hot para las características categóricas (datos de predicción)
datos_prediccion_encoded = pd.get_dummies(datos_prediccion)

# Asegurar que los datos de predicción tengan las mismas columnas que los datos de entrenamiento
missing_cols_prediccion = set(X_train_encoded.columns) - set(datos_prediccion_encoded.columns)
for col in missing_cols_prediccion:
    datos_prediccion_encoded[col] = 0
datos_prediccion_encoded = datos_prediccion_encoded[X_train_encoded.columns]

# Realizar la predicción de los juegos más vendidos para los años 2021 y 2022
prediccion_juegos = best_rf.predict(datos_prediccion_encoded)
print("Predicción de juegos más vendidos para 2021 y 2022:", prediccion_juegos)
