import streamlit as st
import mysql.connector
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA
from datetime import timedelta

# Conectar a la base de datos MySQL y obtener los datos
def get_data():
    conexion = mysql.connector.connect(
        host="localhost",
        user="root",
        password="",
        database="minimarket_tiptop"
    )
    cursor = conexion.cursor(dictionary=True)
    query = "SELECT created_at AS fecha, total_pagar FROM ventas"
    cursor.execute(query)
    records = cursor.fetchall()
    df = pd.DataFrame(records)
    cursor.close()
    conexion.close()
    return df

# Procesar los datos para agrupar por día, semana o mes
def preprocess_data(df, interval='D'):
    df['fecha'] = pd.to_datetime(df['fecha'])
    df.set_index('fecha', inplace=True)
    
    # Convertir total_pagar a float y eliminar valores nulos
    df['total_pagar'] = pd.to_numeric(df['total_pagar'], errors='coerce')
    df.dropna(subset=['total_pagar'], inplace=True)
    
    # Agrupar por el intervalo seleccionado
    df_resampled = df['total_pagar'].resample(interval).sum()
    return df_resampled

# Entrenar el modelo ARIMA y realizar predicciones
def predict_sales(df, periods):
    model = ARIMA(df, order=(1, 1, 1))  # Ajusta el orden según la calidad de predicción deseada
    model_fit = model.fit()
    forecast = model_fit.forecast(steps=periods)
    return forecast

# Configuración de Streamlit
st.title("Predicción de Ventas")
#st.write("Este módulo utiliza el modelo ARIMA para predecir las ventas.")

# Seleccionar el intervalo de predicción
interval = st.selectbox("Selecciona el intervalo de predicción:", ("Día", "Semana", "Mes"))
if interval == "Día":
    interval_code = 'D'
    periods = 7
elif interval == "Semana":
    interval_code = 'W'
    periods = 4
else:
    interval_code = 'M'
    periods = 4

# Cargar y procesar los datos
df = get_data()
df_resampled = preprocess_data(df, interval=interval_code)
st.write("Datos de ventas:")
st.line_chart(df_resampled)

# Predecir las ventas
forecast = predict_sales(df_resampled, periods)
st.write("Predicción de ventas para el siguiente periodo seleccionado:")
st.write(forecast)

# Mostrar resultados de la predicción
st.line_chart(forecast)

