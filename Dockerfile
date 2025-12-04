# Dockerfile simplificado para SD3
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# 1. Copiar requirements primero (para cache de capas)
COPY requirements.txt .

# 2. Instalar dependencias Python
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt && \
    pip install --no-cache-dir xformers

# 3. Crear directorios
RUN mkdir -p /app/models /app/outputs

# 4. Copiar código
COPY src/ /app/src/

# 5. Variables de entorno
ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    TRANSFORMERS_CACHE=/app/models \
    HF_HOME=/app/models \
    TORCH_HOME=/app/models \
    HF_HUB_OFFLINE=0 \
    HF_HUB_DISABLE_TELEMETRY=1

# 6. Comando
CMD ["python", "/app/src/main.py"]