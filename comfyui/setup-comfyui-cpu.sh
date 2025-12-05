#!/bin/bash
# install-comfyui-0376.sh

set -e  # Detener en error

echo "========================================"
echo "Instalando ComfyUI v0.3.76 en CPU"
echo "========================================"

# 1. Crear directorio y entrar
DIR="comfyui-$(date +%Y%m%d_%H%M%S)"
mkdir -p "$DIR" && cd "$DIR"
echo "✓ Directorio creado: $(pwd)"

# 2. Crear Dockerfile
echo "Creando Dockerfile..."
cat > Dockerfile << 'EOF'
FROM python:3.10-slim-bullseye

WORKDIR /ComfyUI

ENV DEBIAN_FRONTEND=noninteractive \
    TZ=UTC \
    USE_CPU=1

RUN apt-get update && apt-get install -y \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 --branch v0.3.76 \
    https://github.com/comfyanonymous/ComfyUI.git /ComfyUI

RUN pip install --no-cache-dir \
    torch==2.1.0 \
    torchvision==0.16.0 \
    torchaudio==2.1.0 \
    --index-url https://download.pytorch.org/whl/cpu

RUN cd /ComfyUI && pip install --no-cache-dir -r requirements.txt

EXPOSE 8188

CMD ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--cpu"]
EOF

echo "✓ Dockerfile creado"

# 3. Crear docker-compose.yml
echo "Creando docker-compose.yml..."
cat > docker-compose.yml << 'EOF'
version: '3.8'

services:
  comfyui:
    build: .
    container_name: comfyui
    restart: unless-stopped
    ports:
      - "8188:8188"
    environment:
      - USE_CPU=1
    volumes:
      - ./models:/ComfyUI/models
      - ./output:/ComfyUI/output
      - ./config:/ComfyUI/config
    command: ["python", "main.py", "--listen", "0.0.0.0", "--port", "8188", "--cpu"]
EOF

echo "✓ docker-compose.yml creado"

# 4. Crear estructura de carpetas
echo "Creando estructura de carpetas..."
mkdir -p models/{checkpoints,loras,vae,embeddings,upscale_models,controlnet}
mkdir -p output/{images,masks,grids}
mkdir -p config custom_nodes input logs

echo "✓ Estructura de carpetas creada"

# 5. Crear archivo de configuración básica
cat > config/extra_model_paths.yaml << 'EOF'
# Configuración de rutas para ComfyUI
base_path: /ComfyUI
checkpoints: models/checkpoints
loras: models/loras
vae: models/vae
embeddings: models/embeddings
upscale_models: models/upscale_models
controlnet: models/controlnet
EOF

# 6. Construir imagen Docker
echo "Construyendo imagen Docker (esto puede tardar 5-10 minutos)..."
docker-compose build --no-cache

# 7. Iniciar contenedor
echo "Iniciando ComfyUI..."
docker-compose up -d

# 8. Esperar y verificar
echo "Esperando a que ComfyUI se inicie..."
sleep 15

# 9. Mostrar información
echo ""
echo "========================================"
echo "INSTALACIÓN COMPLETADA"
echo "========================================"
echo ""
echo "ComfyUI v0.3.76 está corriendo en:"
echo "  → http://localhost:8188"
echo ""
echo "Para tu VPS, también puedes acceder desde:"
echo "  → http://$(curl -s ifconfig.me):8188"
echo ""
echo "ESTRUCTURA DE CARPETAS:"
tree -L 2 .
echo ""
echo "COMANDOS ÚTILES:"
echo "  Ver logs:              docker-compose logs -f"
echo "  Detener:               docker-compose down"
echo "  Reiniciar:             docker-compose restart"
echo "  Actualizar código:     git pull en el repositorio"
echo "  Acceder al contenedor: docker exec -it comfyui bash"
echo ""
echo "========================================"