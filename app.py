# -*- coding: utf-8 -*-
import pandas as pd
import numpy as np
import dash
from dash import dcc, html
import plotly.express as px
from dash.dependencies import Input, Output
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

#%% Cargar datos
df = pd.read_csv("datos_apartamentos_rent.csv", encoding="ISO-8859-1", delimiter=";")

#%% Preprocesamiento
# Asegurar que no haya valores nulos en categorías clave
df["pets_allowed"].fillna("None", inplace=True)
df["category"].fillna("housing/rent/other", inplace=True)

# Eliminar columnas irrelevantes
df_modelamiento = df.drop(columns=["fee","cityname","currency", "price_display", "amenities", "title", "body", "address","source","id", "time","has_photo"])

# Convertir variables categóricas con One-Hot Encoding
X = pd.get_dummies(df_modelamiento.drop(columns=["price"]), 
                   columns=["state", "price_type", "pets_allowed", "category"], drop_first=True)
y = df_modelamiento["price"]

# Rellenar valores nulos con 0
X.fillna(0, inplace=True)

# Dividir en conjunto de entrenamiento y prueba (80% - 20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#%% Entrenar modelo de regresión lineal
modelo = LinearRegression()
modelo.fit(X_train, y_train)

#%% Inicializar la aplicación Dash
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server

# Obtener las variables del modelo
input_variables = X.columns.tolist()

# Extraer opciones de las variables categóricas
states = [col.replace("state_", "") for col in input_variables if col.startswith("state_")]
price_types = [col.replace("price_type_", "") for col in input_variables if col.startswith("price_type_")]
pets_allowed_options = [col.replace("pets_allowed_", "") for col in input_variables if col.startswith("pets_allowed_")]
categories = [col.replace("category_", "") for col in input_variables if col.startswith("category_")]

# Asegurar que todas las opciones necesarias estén en la lista
for pt in ["Monthly", "Weekly", "Monthly|Weekly"]:
    if pt not in price_types:
        price_types.append(pt)

for pet in ["None", "Cats"]:
    if pet not in pets_allowed_options:
        pets_allowed_options.append(pet)

for cat in ["housing/rent/apartment"]:
    if cat not in categories:
        categories.append(cat)

print("Tipos de Precio:", price_types)
print("Mascotas Permitidas:", pets_allowed_options)
print("Categorías:", categories)

#%% Diseño del dashboard
app.layout = html.Div(children=[
    html.H1(children='Tablero de Predicción de Precios de Apartamentos'),

    html.Div(children='''Ingrese los datos del apartamento y haga clic en "Calcular Precio" para obtener una estimación.'''),

    # Controles de entrada
    html.Label("Número de Baños"),
    dcc.Slider(id='bathrooms', min=1, max=10, step=1, value=2, marks={i: str(i) for i in range(1, 11)}),

    html.Label("Número de Habitaciones"),
    dcc.Slider(id='bedrooms', min=1, max=10, step=1, value=3, marks={i: str(i) for i in range(1, 11)}),

    html.Label("Tamaño en pies cuadrados"),
    dcc.Input(id='square_feet', type='number', value=1000, style={'margin-bottom': '20px'}),

    html.Label("Estado"),
    dcc.Dropdown(id='state', options=[{'label': state, 'value': state} for state in states], value='CA'),

    html.Label("Tipo de Precio"),
    dcc.Dropdown(id='price_type', options=[{'label': pt, 'value': pt} for pt in price_types], value='Monthly'),

    html.Label("Mascotas Permitidas"),
    dcc.Dropdown(id='pets_allowed', options=[{'label': pet, 'value': pet} for pet in pets_allowed_options], value='None'),

    html.Label("Categoría"),
    dcc.Dropdown(id='category', options=[{'label': cat, 'value': cat} for cat in categories], value='housing/rent/apartment'),

    html.Button("Calcular Precio", id="submit-button", n_clicks=0, style={'margin-top': '20px'}),

    html.H3(id="prediction-output"),

    dcc.Graph(id="prediction-graph"),  # Ahora el gráfico será un histograma

    html.Div(children='''Variables utilizadas para la predicción:'''),
    html.Ul(id='input-list')
])

#%% Callback para calcular la predicción y actualizar el histograma
@app.callback(
    [Output("prediction-output", "children"),
     Output("prediction-graph", "figure"),
     Output("input-list", "children")],
    [Input("submit-button", "n_clicks")],
    [dash.dependencies.State("bathrooms", "value"),
     dash.dependencies.State("bedrooms", "value"),
     dash.dependencies.State("square_feet", "value"),
     dash.dependencies.State("state", "value"),
     dash.dependencies.State("price_type", "value"),
     dash.dependencies.State("pets_allowed", "value"),
     dash.dependencies.State("category", "value")]
)
def predict_price(n_clicks, bathrooms, bedrooms, square_feet, state, price_type, pets_allowed, category):
    # Inicializar todas las variables en 0
    encoded_input = {var: 0 for var in input_variables}

    # Asignar valores numéricos
    encoded_input['bathrooms'] = bathrooms
    encoded_input['bedrooms'] = bedrooms
    encoded_input['square_feet'] = square_feet

    # Asignar variables categóricas
    for var_name, var_value in [("state", state), ("price_type", price_type), 
                                ("pets_allowed", pets_allowed), ("category", category)]:
        col = f"{var_name}_{var_value}"
        if col in encoded_input:
            encoded_input[col] = 1

    # Crear DataFrame con la estructura correcta
    input_data = pd.DataFrame([encoded_input])
    input_data = input_data.reindex(columns=X.columns, fill_value=0)

    # Hacer la predicción
    predicted_price = modelo.predict(input_data)[0]

    # Definir la columna del estado en el dataset
    state_col = f"state_{state}"

    # Verificar si el estado existe en los datos para evitar errores
    if state_col in df.columns:
        df_state = df[df[state_col] == 1]
    else:
        df_state = df

    # Crear histograma de precios en el estado seleccionado
    fig = px.histogram(
        df_state, x="price",
        title=f"Distribución de Precios en {state}",
        labels={"price": "Precio en USD"},
        nbins=50, opacity=0.7
    )

    # Agregar una línea indicando el precio estimado
    fig.add_vline(x=predicted_price, line_dash="dash", line_color="red", annotation_text="Precio Estimado")

    # Lista de variables utilizadas
    input_list = [html.Li(f"{key}: {value}") for key, value in encoded_input.items() if value != 0]

    return f"Precio Estimado: ${predicted_price:,.2f}", fig, input_list

#%% Ejecutar la aplicación
if __name__ == '__main__':
    app.run_server(debug=True)

