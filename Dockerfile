# Usar imagen con CUDA 12.1 y Python 3.10
FROM nvidia/cuda:12.1.1-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Instalar Python y herramientas básicas
RUN apt-get update && apt-get install -y \
    python3.10 \
    python3.10-venv \
    python3-pip \
    git \
    && rm -rf /var/lib/apt/lists/*

# Crear enlace simbólico para python3
RUN ln -s /usr/bin/python3.10 /usr/bin/python

# Copiar requirements primero
COPY requirements.txt .

# Instalar PyTorch primero con versión específica
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    torch==2.9.0 \
    torchvision==0.20.0 \
    torchaudio==2.9.0 \
    --index-url https://download.pytorch.org/whl/cu121

RUN pip install --no-cache-dir -r requirements.txt
# Instalar xformers con la versión correcta
RUN pip install --no-cache-dir xformers==0.0.33.post1

# Instalar el resto de dependencias
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
