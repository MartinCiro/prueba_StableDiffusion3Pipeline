FROM python:3.10-slim-bullseye

WORKDIR /ComfyUI

# Fijar versiones específicas para compatibilidad
ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    USE_CPU=1 \
    # Forzar versión antigua de numpy
    NUMPY_VERSION=1.26.4

# Instalar dependencias del sistema
RUN apt-get update && apt-get install -y \
    git \
    curl \
    wget \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Clonar ComfyUI v0.3.76
RUN git clone --depth 1 --branch v0.3.76 \
    https://github.com/comfyanonymous/ComfyUI.git /ComfyUI

# Instalar numpy primero con versión específica
RUN pip install --no-cache-dir \
    numpy==${NUMPY_VERSION} \
    pillow==10.1.0 \
    scipy==1.11.4 \
    scikit-image==0.22.0

# Instalar PyTorch 2.1.0 para CPU con dependencias específicas
RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

# Instalar transformers versión compatible con PyTorch 2.1.0
RUN pip install --no-cache-dir \
    transformers==4.35.2 \
    accelerate==0.25.0

# Instalar requirements.txt de ComfyUI con override de versiones
RUN cd /ComfyUI && \
    pip install --no-cache-dir \
    -r requirements.txt \
    --no-deps  # No instalar dependencias automáticas

# Instalar dependencias adicionales con versiones específicas
RUN pip install --no-cache-dir \
    opencv-python-headless==4.8.1.78 \
    safetensors==0.4.2 \
    aiohttp==3.9.1 \
    requests==2.31.0 \
    protobuf==3.20.3 \
    psutil==5.9.6 \
    pyyaml==6.0.1 \
    packaging==23.2 \
    typing-extensions==4.8.0 \
    tqdm==4.66.1

# Verificar versiones instaladas
RUN python -c "import numpy; print(f'numpy: {numpy.__version__}')" && \
    python -c "import torch; print(f'torch: {torch.__version__}')" && \
    python -c "import transformers; print(f'transformers: {transformers.__version__}')"

EXPOSE 8188

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--cpu", "--disable-xformers"]