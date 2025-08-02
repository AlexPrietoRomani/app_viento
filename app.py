# app.py

import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
import joblib
import json
import plotly.graph_objects as go
import os
import random

# ===================================================================
# Configuración de la Página y Título
# ===================================================================
# Configuración inicial de la página (debe ser el primer comando de Streamlit)
st.set_page_config(
    page_title="AgroClima Viento | Pronóstico LSTM",
    page_icon="🌬️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Barra Lateral (Sidebar) ---
with st.sidebar:
    st.image("https://www.freeiconspng.com/uploads/weather-icon-png-16.png", width=100) # Reemplazar con un logo si se tiene
    st.title("AgroClima Viento")
    st.info(
        "Esta aplicación utiliza un modelo de Deep Learning (LSTM) para generar un pronóstico de alta resolución "
        "de la velocidad del viento para las próximas 24 horas."
    )
    st.success(
        "**Beneficios:**\n"
        "- **Optimizar** ventanas de pulverización.\n"
        "- **Reducir** la deriva de fitosanitarios.\n"
        "- **Mejorar** la seguridad en operaciones agrícolas."
    )
    st.caption("Desarrollado por Alex Anthony Prieto Romani")
    st.caption("Contacto: alexprieto1997@gmail.com")

# --- Cuerpo Principal de la Aplicación ---

# Título Principal
st.title("🌬️ Pronóstico Diario de Viento con Análisis de Incertidumbre")
st.markdown("---")

# ===================================================================
# Funciones Auxiliares (Carga, Preprocesamiento, Predicción)
# ===================================================================

@st.cache_resource
def load_artifacts(model_path, scaler_path, config_path):
    """
    Carga y cachea los artefactos del modelo desde el disco.

    Esta función es decorada con @st.cache_resource para asegurar que los modelos
    y objetos pesados se carguen en memoria una sola vez, mejorando el
    rendimiento de la aplicación en recargas posteriores.

    Args:
        model_path (str): Ruta al archivo del modelo Keras guardado (ej. 'modelo.keras').
        scaler_path (str): Ruta al archivo del objeto Scaler guardado (ej. 'scaler.joblib').
        config_path (str): Ruta al archivo de configuración JSON (ej. 'config.json').

    Returns:
        tuple: Una tupla conteniendo (model, scaler, config) si la carga es exitosa.
               En caso de error, devuelve (None, None, None).
    """
    try:
        model = tf.keras.models.load_model(model_path, compile=False)
        scaler = joblib.load(scaler_path)
        with open(config_path, "r") as f:
            config = json.load(f)
        return model, scaler, config
    except Exception as e:
        st.error(f"Error crítico al cargar los artefactos del modelo: {e}. Asegúrese de que los archivos estén en el directorio correcto.")
        return None, None, None

def preprocess_data(df_raw, expected_features):
    """
    Transforma un DataFrame de datos meteorológicos brutos al formato requerido por el modelo.

    Realiza las siguientes operaciones clave:
    1.  Asegura que las columnas numéricas tengan el tipo de dato correcto.
    2.  Convierte y combina las columnas 'Date' y 'hora' en un índice Datetime.
    3.  Maneja valores nulos que puedan surgir de la conversión de tipos.
    4.  Crea características cíclicas ('hora_sin', 'hora_cos') para la hora del día.
    5.  Selecciona y reordena las columnas para que coincidan con 'expected_features'.

    Args:
        df_raw (pd.DataFrame): DataFrame con los datos en su formato original.
        expected_features (list): Lista de strings con los nombres de las columnas
                                  y el orden que el modelo espera.

    Returns:
        pd.DataFrame: El DataFrame procesado y listo para ser escalado.
    """
    df = df_raw.copy()
    
    # --- Robustez de Tipos de Datos ---
    # Forzar la conversión de las columnas numéricas. 'coerce' convierte errores en NaN.
    numeric_cols = ['temperatura media', 'humedad', 'viento', 'lluvia']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # Intentar parsear las fechas, manejando múltiples formatos si es necesario.
    try:
        df['Date'] = pd.to_datetime(df['Date'])
    except Exception:
        df['Date'] = pd.to_datetime(df['Date'], dayfirst=True)
        
    df['datetime'] = pd.to_datetime(df['Date'].astype(str) + ' ' + df['hora'].astype(str))
    df = df.set_index('datetime').sort_index()
    
    # --- Comprobación de Nulos después de la conversión ---
    if df[numeric_cols].isnull().sum().sum() > 0:
        st.warning("Se detectaron valores no numéricos en las columnas de datos. Se han convertido a nulos. Considera imputarlos o revisar tu CSV.")
        # Llenar NaNs con el método 'forward fill' como estrategia simple
        df.fillna(method='ffill', inplace=True)
        df.fillna(method='bfill', inplace=True) # bfill por si el primer valor es NaN

    df['hora_sin'] = np.sin(2 * np.pi * (df.index.hour + df.index.minute / 60) / 24)
    df['hora_cos'] = np.cos(2 * np.pi * (df.index.hour + df.index.minute / 60) / 24)
    
    return df[expected_features]

def iterative_mc_dropout_forecast(
    initial_sequence: np.ndarray, 
    future_exog: pd.DataFrame, 
    model: tf.keras.Model, 
    scaler: object, 
    features_order: list, 
    steps: int = 48, 
    mc_iterations: int = 50
) -> tuple:
    """
    Genera un pronóstico de múltiples pasos con intervalos de confianza.

    Utiliza una estrategia iterativa (autorregresiva) donde la predicción de un
    paso se utiliza como entrada para el siguiente. La incertidumbre se cuantifica
    mediante la técnica de Monte Carlo Dropout, ejecutando el pronóstico múltiples
    veces con las capas de Dropout activadas.

    Args:
        initial_sequence (np.ndarray): Array de NumPy con los datos históricos ya
                                       escalados y con el formato (lookback, n_features).
        future_exog (pd.DataFrame): DataFrame con las variables exógenas futuras ya
                                    escaladas. No incluye la columna objetivo ('viento').
        model (tf.keras.Model): El modelo LSTM entrenado.
        scaler (object): El objeto MinMaxScaler ajustado para des-escalar los resultados.
        features_order (list): El orden exacto de las columnas del modelo.
        steps (int, optional): Número de pasos a predecir en el futuro. Por defecto 48.
        mc_iterations (int, optional): Número de iteraciones para Monte Carlo Dropout.
                                       Por defecto 50.

    Returns:
        tuple: Una tupla con tres arrays de NumPy: (mean_forecast, lower_bound, upper_bound).
    """
    all_forecasts = []
    
    # Bucle principal de Monte Carlo
    for _ in range(mc_iterations):
        current_sequence = initial_sequence.astype(np.float32) 
        single_run_forecasts = []
        
        # Bucle de predicción iterativa para cada paso futuro
        for j in range(steps):
            input_reshaped = np.reshape(current_sequence, (1, current_sequence.shape[0], current_sequence.shape[1]))
            
            # Realizar predicción con Dropout activado para cuantificar incertidumbre
            prediction_scaled = model(input_reshaped, training=True)
            single_run_forecasts.append(prediction_scaled[0, 0])
            
            # Construir la entrada para el siguiente paso
            next_step_exog = future_exog.iloc[[j]]
            next_step_features_df = next_step_exog.copy()
            next_step_features_df['viento'] = prediction_scaled[0, 0]
            
            # Forzar el tipo de dato y apilar para la siguiente iteración
            next_step_values = next_step_features_df[features_order].values.astype(np.float32)
            current_sequence = np.vstack([current_sequence[1:], next_step_values])
            
        all_forecasts.append(single_run_forecasts)
        
    # Procesar los resultados de todas las ejecuciones de Monte Carlo
    all_forecasts_np = np.array(all_forecasts, dtype=np.float32)
    
    # Des-escalar todas las predicciones a la vez para mayor eficiencia
    temp_descale = np.zeros((all_forecasts_np.size, len(features_order)))
    temp_descale[:, 0] = all_forecasts_np.flatten()
    forecasts_descaled = scaler.inverse_transform(temp_descale)[:, 0].reshape(mc_iterations, steps)
    
    # Calcular estadísticas finales: media e intervalos de confianza
    mean_forecast = forecasts_descaled.mean(axis=0)
    lower_bound = np.percentile(forecasts_descaled, 2.5, axis=0)
    upper_bound = np.percentile(forecasts_descaled, 97.5, axis=0)
    
    return mean_forecast, lower_bound, upper_bound

def iterative_deterministic_forecast(
    initial_sequence: np.ndarray, 
    future_exog: pd.DataFrame, 
    model: tf.keras.Model, 
    scaler: object, 
    features_order: list, 
    steps: int = 48
) -> np.ndarray:
    """
    Genera un pronóstico determinista de múltiples pasos.

    Utiliza una estrategia iterativa (autorregresiva) pero sin activar el
    Dropout, resultando en una única predicción sin intervalos de confianza.

    Args:
        initial_sequence (np.ndarray): Datos históricos escalados.
        future_exog (pd.DataFrame): Variables exógenas futuras escaladas.
        model (tf.keras.Model): El modelo entrenado.
        scaler (object): El objeto MinMaxScaler para des-escalar.
        features_order (list): El orden de las columnas.
        steps (int, optional): Pasos a predecir. Por defecto 48.

    Returns:
        np.ndarray: Un array de NumPy con el pronóstico de viento.
    """
    current_sequence = initial_sequence.astype(np.float32)
    forecasts_scaled = []
    
    for j in range(steps):
        input_reshaped = np.reshape(current_sequence, (1, current_sequence.shape[0], current_sequence.shape[1]))
        
        prediction_scaled = model(input_reshaped, training=False)
        
        forecasts_scaled.append(prediction_scaled[0, 0])
        
        next_step_exog = future_exog.iloc[[j]]
        next_step_features_df = next_step_exog.copy()
        next_step_features_df['viento'] = prediction_scaled[0, 0]
        
        next_step_values = next_step_features_df[features_order].values.astype(np.float32)
        current_sequence = np.vstack([current_sequence[1:], next_step_values])
        
    # Des-escalar el único pronóstico generado
    forecasts_scaled_np = np.array(forecasts_scaled, dtype=np.float32)
    temp_descale = np.zeros((forecasts_scaled_np.size, len(features_order)))
    temp_descale[:, 0] = forecasts_scaled_np.flatten()
    forecasts_descaled = scaler.inverse_transform(temp_descale)[:, 0]
    
    return forecasts_descaled

# ===================================================================
# FIJAR SEMILLAS PARA REPRODUCIBILIDAD
# ===================================================================
# Esto es crucial para que los resultados de Monte Carlo Dropout sean
# consistentes entre diferentes ejecuciones y entornos (CPU vs. GPU).
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
# Opcional: para una reproducibilidad aún más estricta en GPU
# os.environ['TF_DETERMINISTIC_OPS'] = '1

# ==============================================================================
# SECCIÓN 2: CARGA INICIAL Y CONFIGURACIÓN DE LA APLICACIÓN
# ==============================================================================
# Se cargan los artefactos del modelo una sola vez al iniciar la aplicación
# gracias al decorador @st.cache_resource en la función load_artifacts.
model, scaler, config = load_artifacts(
    "Modelos/modelo_viento_completo.keras", 
    "Modelos/scaler_viento.joblib",
    "Modelos/model_config.json"
)

# El código principal de la aplicación solo se ejecuta si la carga de artefactos fue exitosa.
if model is not None:
    # Extraer constantes de configuración para fácil acceso
    LOOKBACK = config['lookback']
    FEATURES = config['features']
    RAW_FEATURES_HIST = ['Date', 'hora', 'temperatura media', 'humedad', 'viento', 'lluvia']
    
    # ==============================================================================
    # SECCIÓN 3: DISEÑO DE LA INTERFAZ DE USUARIO (UI)
    # ==============================================================================
    st.header("Paso Único: Cargar Archivo de Datos Históricos")
    st.markdown(f"Sube un archivo CSV con al menos los últimos **{LOOKBACK//2} horas ({LOOKBACK} registros)** de datos.")
    
    # --- Layout en columnas para la introducción y las instrucciones ---
    col1, col2 = st.columns((2, 1.5), gap="large")

    with col1:
        st.subheader("Bienvenido a la Herramienta de Pronóstico Inteligente")
        st.markdown(
        """
        Obtenga una predicción detallada de la velocidad del viento para el día siguiente,
        generada por un modelo de series temporales avanzado (LSTM). Nuestra herramienta no solo
        le dice qué esperar, sino también **qué tan confiable es el pronóstico**, mostrándole
        los posibles rangos de variación.
        """
        )
        st.markdown("#### ¿Cómo Empezar? Siga estos 3 sencillos pasos:")
        st.markdown(
            """
            1.  **Prepare su Archivo**: Asegúrese de tener un archivo CSV con sus datos meteorológicos históricos.
            2.  **Cárguelo Abajo**: Use el cargador de archivos para subir su historial.
            3.  **Genere el Pronóstico**: Haga clic en el botón para visualizar la predicción de 24 horas.
            """
        )

    with col2:
        st.subheader("Formato del Archivo Requerido")
        # Usar la variable LOOKBACK cargada para que sea consistente
        st.info(f"El archivo CSV debe contener al menos **{LOOKBACK} registros** ({LOOKBACK//2} horas) de datos.")
        
        with st.expander("Ver ejemplo del formato de columnas"):
            st.code(
            """
# Ejemplo de las primeras filas de su archivo .csv
Date,hora,temperatura media,humedad,viento,lluvia
2024-01-01,00:30:00,21.8,81,4.8,0.0
2024-01-01,01:00:00,21.7,81,4.8,0.0
2024-01-01,01:30:00,21.4,82,4.8,0.0
...
            """,
                language="csv"
            )
            
    st.markdown("---")
    
    # --- Widget para la carga de archivos ---
    st.header("Paso Único: Cargar Archivo de Datos Históricos")
    
    # Widget para la carga de archivos. El usuario interactúa aquí.
    uploaded_hist_file = st.file_uploader("Historial Meteorológico (CSV)", type="csv")

    # La lógica de predicción se activa solo si el usuario ha subido un archivo.
    if uploaded_hist_file:
        try:
            # Leer el archivo CSV en un DataFrame de pandas.
            # Se especifica el punto como separador decimal para evitar errores de localización.
            hist_df_raw = pd.read_csv(uploaded_hist_file, decimal='.')

            # --- Validación de Entrada del Usuario ---
            # Se comprueba si el archivo tiene suficientes datos para el lookback del modelo.
            if len(hist_df_raw) < LOOKBACK:
                st.error(f"El archivo histórico debe tener al menos {LOOKBACK} filas. El archivo subido solo tiene {len(hist_df_raw)}.")
            else:
                # Botón principal que inicia todo el proceso de pronóstico.
                if st.button("🌦️ Generar Pronóstico de 24 Horas", use_container_width=True):
                    with st.spinner("Realizando pronóstico iterativo... Esto puede tardar un minuto."):
                        
                        # ==============================================================================
                        # SECCIÓN 4: LÓGICA DE PROCESAMIENTO Y PREDICCIÓN
                        # ==============================================================================
                        
                        # --- Paso 4.1: Preprocesamiento del Historial ---
                        # Se transforman los datos brutos cargados al formato que el modelo espera.
                        hist_processed = preprocess_data(hist_df_raw, FEATURES)
                        
                        # Verificación final de nulos antes de proceder.
                        if hist_processed.isnull().sum().sum() > 0:
                            st.error("Los datos contienen valores nulos incluso después de la limpieza. No se puede continuar.")
                        else:
                            # --- Paso 4.2: Generación Automática de Exógenas Futuras ---
                            # Se aplica una heurística de persistencia para estimar las variables futuras.
                            last_24h_hist = hist_processed.tail(48)
                            last_timestamp = last_24h_hist.index[-1]
                            
                            # Se crea un rango de fechas para las próximas 24h (48 pasos de 30 min).
                            future_dates = pd.date_range(start=last_timestamp + pd.Timedelta(minutes=30), periods=48, freq='30min')
                            
                            # Se asume que las condiciones de temp, humedad y lluvia de mañana serán similares a las de hoy.
                            future_df_generated = pd.DataFrame(index=future_dates)
                            future_df_generated['temperatura media'] = last_24h_hist['temperatura media'].values
                            future_df_generated['humedad'] = last_24h_hist['humedad'].values
                            future_df_generated['lluvia'] = last_24h_hist['lluvia'].values
                            future_df_generated['viento'] = 0 # Placeholder que será reemplazado en la predicción.
                            
                            # Se calculan las características cíclicas para las fechas futuras.
                            future_df_generated['hora_sin'] = np.sin(2 * np.pi * (future_df_generated.index.hour + future_df_generated.index.minute / 60) / 24)
                            future_df_generated['hora_cos'] = np.cos(2 * np.pi * (future_df_generated.index.hour + future_df_generated.index.minute / 60) / 24)
                            
                            future_processed = future_df_generated[FEATURES]
                            
                            # --- Paso 4.3: Preparación de Secuencias para el Modelo ---
                            initial_sequence_scaled = scaler.transform(hist_processed.tail(LOOKBACK)).astype(np.float32)
                            future_exog_scaled = pd.DataFrame(scaler.transform(future_processed), columns=FEATURES).astype(np.float32)

                            # --- Paso 4.4: Ejecución del Pronóstico ---
                            # Se llama a la función principal que contiene la lógica de predicción iterativa.
                            mean_fc, lower_fc, upper_fc = iterative_mc_dropout_forecast(
                                initial_sequence_scaled, future_exog_scaled, model, scaler, FEATURES, steps=48, mc_iterations=50
                            )
                            
                            # Llamamos a la nueva función sin predicción iterativa
                            #mean_fc = iterative_deterministic_forecast(
                            #    initial_sequence_scaled, future_exog_scaled, model, scaler, FEATURES, steps=48
                            #)
                            
                            # ==============================================================================
                            # SECCIÓN 5: ANÁLISIS Y VISUALIZACIÓN DE RESULTADOS
                            # ==============================================================================
                            
                            # --- Paso 5.1: Consolidar Resultados en un DataFrame ---
                            results_df = pd.DataFrame({
                                'Pronóstico Viento': mean_fc,
                                'Viento Límite Inferior': lower_fc,
                                'Viento Límite Superior': upper_fc,
                                'Temperatura Estimada': future_processed['temperatura media'].values
                            }, index=future_processed.index)

                            # --- Paso 5.2: Definir Criterios y Identificar Ventanas Óptimas ---
                            VIENTO_MAX_OPTIMO = 10.0
                            TEMP_MAX_OPTIMA = 30.0
                            
                            # Versión con limites de viento y temperatura
                            results_df['Condición Óptima'] = (results_df['Viento Límite Superior'] < VIENTO_MAX_OPTIMO) & \
                                                        (results_df['Temperatura Estimada'] < TEMP_MAX_OPTIMA)

                            # Versión sin limites de viento y temperatura
                            results_df['Condición Óptima'] = (results_df['Pronóstico Viento'] < VIENTO_MAX_OPTIMO) & \
                                        (results_df['Temperatura Estimada'] < TEMP_MAX_OPTIMA)
                                    
                            # --- Paso 5.3: Resumen Ejecutivo ---
                            st.header("Dashboard de Decisión Agronómica")
                            col1, col2, col3 = st.columns(3)
                            
                            #viento_max_esperado = results_df['Viento Límite Superior'].max() 
                            viento_max_esperado = results_df['Pronóstico Viento'].max()
                            col1.metric(label="💨 Viento Máximo Esperado", value=f"{viento_max_esperado:.1f} km/h", help="El valor más alto que podría alcanzar el viento.")
                            
                            horas_optimas = results_df['Condición Óptima'].sum() * 0.5
                            col2.metric(label="✅ Horas Óptimas para Aplicación", value=f"{horas_optimas:.1f} horas", help=f"Viento < {VIENTO_MAX_OPTIMO} km/h y Temp < {TEMP_MAX_OPTIMA}°C.")
                            
                            avg_interval_width = (results_df['Viento Límite Superior'] - results_df['Viento Límite Inferior']).mean()
                            col3.metric(label="📊 Fiabilidad del Pronóstico", value=f"{avg_interval_width:.2f} km/h", help="Amplitud promedio del intervalo de confianza del viento.")

                            # --- Paso 5.4: Lógica para Agrupar Ventanas Contiguas ---
                            def get_optimal_blocks(condition_series):
                                """Encuentra bloques de tiempo contiguos donde la condición es True."""
                                blocks = []
                                in_block = False
                                start_time = None
                                for timestamp, is_optimal in condition_series.items():
                                    if is_optimal and not in_block:
                                        in_block = True
                                        start_time = timestamp
                                    elif not is_optimal and in_block:
                                        in_block = False
                                        end_time = timestamp - pd.Timedelta(minutes=30) # El bloque terminó en el paso anterior
                                        blocks.append((start_time, end_time))
                                if in_block: # Si el último período es óptimo, cerrar el bloque
                                    blocks.append((start_time, condition_series.index[-1]))
                                return blocks

                            optimal_blocks = get_optimal_blocks(results_df['Condición Óptima'])

                            # --- Paso 5.5: Gráficos Duales de Viento y Temperatura ---
                            st.subheader("Visualización del Pronóstico y Ventanas de Aplicación")

                            if not optimal_blocks:
                                st.warning("⚠️ No se encontraron ventanas de aplicación óptimas para las próximas 24 horas según los criterios definidos.")
    
                            # Gráfico de Viento
                            fig_viento = go.Figure()
                            
                            # Añadir la línea de limite de viento superior e inferior
                            fig_viento.add_trace(go.Scatter(x=results_df.index, y=results_df['Viento Límite Superior'], mode='lines', line=dict(width=0), showlegend=False))
                            fig_viento.add_trace(go.Scatter(x=results_df.index, y=results_df['Viento Límite Inferior'], mode='lines', line=dict(width=0), fill='tonexty', fillcolor='rgba(255, 165, 0, 0.2)', name='Incertidumbre (95%)'))
                            
                            # Añadir la línea de pronóstico de viento
                            fig_viento.add_trace(go.Scatter(x=results_df.index, y=results_df['Pronóstico Viento'], mode='lines+markers', line=dict(color='orangered', width=3), name='Pronóstico Viento'))
                            
                            # Linea horizontal para el límite óptimo de viento
                            fig_viento.add_hline(y=VIENTO_MAX_OPTIMO, line_dash="dot", line_color="red", annotation_text=f"Límite Viento ({VIENTO_MAX_OPTIMO} km/h)", annotation_position="bottom right")

                            # Gráfico de Temperatura
                            fig_temp = go.Figure()
                            
                            # Añadir la línea de pronóstico de temperatura
                            fig_temp.add_trace(go.Scatter(x=results_df.index, y=results_df['Temperatura Estimada'], mode='lines+markers', line=dict(color='deepskyblue', width=3), name='Temperatura Estimada'))
                            
                            # Línea horizontal para el límite óptimo de temperatura
                            fig_temp.add_hline(y=TEMP_MAX_OPTIMA, line_dash="dot", line_color="red", annotation_text=f"Límite Temp ({TEMP_MAX_OPTIMA}°C)", annotation_position="bottom right")

                            # Añadir los bloques de ventana óptima a AMBOS gráficos
                            if optimal_blocks:
                                for start, end in optimal_blocks:
                                    # Ajuste para que el rectángulo cubra todo el intervalo de 30 min
                                    rect_start = start - pd.Timedelta(minutes=15)
                                    rect_end = end + pd.Timedelta(minutes=15)
                                    fig_viento.add_vrect(x0=rect_start, x1=rect_end, fillcolor="green", opacity=0.15, line_width=0, annotation_text="Óptimo", annotation_position="top left")
                                    fig_temp.add_vrect(x0=rect_start, x1=rect_end, fillcolor="green", opacity=0.15, line_width=0, showlegend=False)

                            fig_viento.update_layout(title='Pronóstico de Viento', yaxis_title='Viento (km/h)', legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.6)'))
                            fig_temp.update_layout(title='Pronóstico de Temperatura', yaxis_title='Temperatura (°C)', legend=dict(x=0.01, y=0.99, bgcolor='rgba(255,255,255,0.6)'))

                            # Sincronizar los ejes x
                            fig_viento.update_xaxes(matches='x')
                            fig_temp.update_xaxes(matches='x')

                            st.plotly_chart(fig_viento, use_container_width=True)
                            st.plotly_chart(fig_temp, use_container_width=True)

                            # ==============================================================================
                            # SECCIÓN 6: TABLA DE DATOS Y ANÁLISIS POR TURNO
                            # ==============================================================================
                            st.subheader("Detalle del Pronóstico Horario")
                            
                            # Formatear la tabla para mostrarla
                            display_df = results_df[['Pronóstico Viento', 'Temperatura Estimada', 'Condición Óptima']].copy()
                            display_df.index = display_df.index.strftime('%H:%M')
                            display_df.rename(columns={
                                'Pronóstico Viento': 'Viento (km/h)',
                                'Temperatura Estimada': 'Temp (°C)',
                                'Condición Óptima': '¿Es Óptimo?'
                            }, inplace=True)
                            display_df['Viento (km/h)'] = display_df['Viento (km/h)'].round(1)
                            display_df['Temp (°C)'] = display_df['Temp (°C)'].round(1)
                            display_df['¿Es Óptimo?'] = display_df['¿Es Óptimo?'].map({True: '✅ Sí', False: '❌ No'})
                            
                            st.dataframe(display_df.T, use_container_width=True) # Transponer para mejor visualización en móvil
                        
        # Manejo de errores genérico para cualquier problema durante la ejecución.
        except Exception as e:
            st.error(f"Ha ocurrido un error inesperado: {e}. Revise el formato de su archivo CSV.")