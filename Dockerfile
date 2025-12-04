# Dockerfile corregido
FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir torch==2.9.0 --index-url https://download.pytorch.org/whl/cpu

# VERSIONES COMPATIBLES
RUN pip install --no-cache-dir \
    transformers==4.46.0 \  
    huggingface-hub==1.1.7 \
    accelerate==0.30.1 \
    Pillow==10.3.0 \
    redis==5.0.6 \
    psutil==5.9.8 \
    python-dotenv==1.0.1

# Instalar diffusers desde git
RUN pip install --no-cache-dir git+https://github.com/huggingface/diffusers.git

RUN mkdir -p /app/models /app/outputs /app/src
COPY src/ /app/src/

ENV PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/src \
    DEVICE=cpu

CMD ["python", "/app/src/main.py"]