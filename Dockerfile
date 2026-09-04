# Imagen del Copiloto Power BI. Ollama va en su propio contenedor (ver
# docker-compose.yml) — la app le habla por la red interna de compose, via
# "http://ollama:11434/v1".
FROM python:3.10-slim

WORKDIR /app

# pandas/pyarrow/pbixray a veces necesitan compilar alguna dependencia nativa
# segun la plataforma — build-essential lo cubre sin inflar demasiado la imagen.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
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
