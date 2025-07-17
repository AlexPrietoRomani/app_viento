Asegúrese de tener Docker Desktop (o Docker Engine en Linux) instalado y en ejecución en su máquina.

1.  **Abra una nueva terminal** (puede ser PowerShell, CMD o la terminal de su sistema operativo, no necesita estar en el entorno Conda).
2.  **Navegue a la carpeta de su proyecto:**
    ```bash
    cd C:\usuario\app_viento
    ```

3.  **Construya la imagen de Docker:**
    *   El comando `docker build` lee el `Dockerfile`.
    *   `-t viento-pronostico-app` le da un nombre (una "etiqueta") a su imagen para que sea fácil de encontrar.
    *   `.` al final le dice a Docker que el contexto de construcción (donde se encuentran el `Dockerfile` y los archivos de la app) es la carpeta actual.
    ```bash
    docker build -t viento-pronostico-app .
    ```
    La primera vez que ejecute esto, Docker descargará la imagen de Python y luego instalará todas las dependencias del `requirements.txt`. Esto puede tardar varios minutos. Las siguientes veces que construya (si solo cambia el `app.py`), será mucho más rápido gracias al sistema de caché de Docker.

4.  **Ejecute el contenedor a partir de la imagen:**
    *   `docker run` inicia un nuevo contenedor.
    *   `--rm` es una buena práctica: elimina el contenedor automáticamente cuando lo detiene.
    *   `-p 8501:8501` mapea el puerto 8501 de su máquina al puerto 8501 dentro del contenedor. Esto le permite acceder a la app desde su navegador.
    *   `viento-pronostico-app` es el nombre de la imagen que queremos ejecutar.
    ```bash
    docker run --rm -p 8501:8501 viento-pronostico-app
    ```

5.  **Acceda a la aplicación:**
    *   Abra su navegador web y vaya a: `http://localhost:8501`
    *   ¡Verá su aplicación de Streamlit funcionando perfectamente, aislada dentro de su contenedor Docker!

### **Para Compartir con Otros**

Ahora, para que otra persona ejecute su aplicación, solo necesita:
1.  Tener Docker instalado.
2.  Recibir la carpeta completa de su proyecto (`app_viento`).
3.  Ejecutar los mismos dos comandos `docker build` y `docker run` en su propia terminal.

Ha creado un artefacto de software portátil y autocontenido, resolviendo de forma definitiva todos los problemas de "en mi máquina funciona".