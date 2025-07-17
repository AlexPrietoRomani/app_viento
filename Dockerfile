# ===================================================================
# Fase 1: Imagen base oficial de Python
# Usamos una imagen de Python 3.9 que es muy estable con TensorFlow 2.x
# ===================================================================
FROM python:3.10-slim

# ===================================================================
# Metadatos y configuración del entorno
# ===================================================================
# Establecer el directorio de trabajo dentro del contenedor
WORKDIR /app

# Prevenir que Python genere archivos .pyc y los escriba en el disco
ENV PYTHONDONTWRITEBYTECODE 1
# Asegurar que la salida de Python no se almacene en búfer
ENV PYTHONUNBUFFERED 1

# ===================================================================
# Instalación de dependencias del sistema (si fueran necesarias)
# En este caso, no necesitamos librerías extra del sistema operativo
# ===================================================================
# RUN apt-get update && apt-get install -y ...

# ===================================================================
# Instalación de las dependencias de Python
# ===================================================================
# Copiar primero el archivo de requerimientos para aprovechar el caché de Docker
COPY requirements.txt .

# Instalar las dependencias usando pip
# --no-cache-dir para no guardar la caché y mantener la imagen ligera
# --upgrade pip para tener la última versión de pip
RUN pip install --no-cache-dir --upgrade pip -r requirements.txt

# ===================================================================
# Copiar los archivos de la aplicación al contenedor
# ===================================================================
# Copiar todos los archivos de la carpeta actual al directorio /app del contenedor
COPY . .

# ===================================================================
# Exponer el puerto y definir el comando de ejecución
# ===================================================================
# Exponer el puerto que Streamlit usa por defecto
EXPOSE 8501

# Definir la variable de salud del contenedor para que sepa cuándo está listo
# Streamlit tiene un endpoint de salud incorporado
HEALTHCHECK CMD curl --fail http://localhost:8501/_stcore/health

# El comando que se ejecutará cuando el contenedor se inicie
# Le decimos a Streamlit que se ejecute en todas las interfaces de red (0.0.0.0)
# y deshabilitamos la apertura automática del navegador.
ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]