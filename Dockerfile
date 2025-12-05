FROM python:3.10-slim

WORKDIR /app

# AÑADIR sentencepiece al apt-get
RUN apt-get update && apt-get install -y \
    git \
    libsentencepiece-dev \ 
    pkg-config \ 
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir --upgrade pip wheel setuptools

COPY requirements.txt .

# Instalar PyTorch y dependencias CRÍTICAS
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch==2.9.0 && \
    pip install --no-cache-dir \
    protobuf==6.30.0 \
    sentencepiece \ 
    transformers

RUN pip install --no-cache-dir \
    huggingface-hub \
    accelerate \
    Pillow \
    redis \
    psutil \
    python-dotenv \
    diffusers

RUN mkdir -p /app/models /app/outputs /app/src
COPY src/ /app/src/

# Corregir código problemático automáticamente
RUN sed -i "s/retry_on_timeout=True,//g" /app/src/main.py && \
    sed -i "s/safety_checker=None,//g" /app/src/main.py && \
    sed -i "s/torch_dtype=/dtype=/g" /app/src/main.py

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DEVICE=cpu \
    HF_HOME=/app/models \
    TRANSFORMERS_CACHE=/app/models

CMD ["python", "/app/src/main.py"]