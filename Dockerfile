# Imagen del Copiloto Power BI. Ollama va en su propio contenedor (ver
# docker-compose.yml) — la app le habla por la red interna de compose, via
# "http://ollama:11434/v1".
FROM python:3.10-slim

WORKDIR /app

# pandas/pyarrow/pbixray a veces necesitan compilar alguna dependencia nativa
# segun la plataforma — build-essential lo cubre sin inflar demasiado la imagen.
# unixodbc-dev + el driver de Microsoft son para pyodbc (login contra
# COE.User_Admin en SQL Server, ver app/db.py) — sin esto pyodbc.connect()
# no encuentra ningun driver y el login falla con un error de conexion.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl gnupg2 unixodbc unixodbc-dev \
    && . /etc/os-release \
    && curl -sSL https://packages.microsoft.com/keys/microsoft.asc | gpg --dearmor -o /usr/share/keyrings/microsoft-prod.gpg \
    && curl -sSL "https://packages.microsoft.com/config/debian/${VERSION_ID}/prod.list" > /etc/apt/sources.list.d/mssql-release.list \
    && sed -i 's#deb #deb [signed-by=/usr/share/keyrings/microsoft-prod.gpg] #' /etc/apt/sources.list.d/mssql-release.list \
    && apt-get update \
    && ACCEPT_EULA=Y apt-get install -y --no-install-recommends msodbcsql18 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app/ ./app/

# tablero/ (el .pbix) y data/ (el cache extraido) se montan como volumenes en
# tiempo de ejecucion, NO se copian al build — asi el .pbix de cada instalacion
# no queda horneado en la imagen, y "Actualizar datos" persiste entre reinicios
# del contenedor.
RUN mkdir -p tablero data/pbix

EXPOSE 8011

# --host 0.0.0.0 es obligatorio: sin esto, uvicorn solo escucha en localhost
# DENTRO del contenedor y no seria alcanzable desde afuera aunque el puerto
# este publicado.
CMD ["python", "-m", "uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8011"]
