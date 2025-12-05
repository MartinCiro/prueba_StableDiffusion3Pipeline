#!/bin/bash

# Configurar variables para optimización CPU
export OMP_NUM_THREADS=${OMP_NUM_THREADS:-4}
export MKL_NUM_THREADS=${MKL_NUM_THREADS:-4}
export NUMEXPR_NUM_THREADS=${NUMEXPR_NUM_THREADS:-4}
export OPENBLAS_NUM_THREADS=${OPENBLAS_NUM_THREADS:-4}

echo "=== ComfyUI CPU Optimizado ==="
echo "Threads configurados:"
echo "  OMP_NUM_THREADS: $OMP_NUM_THREADS"
echo "  MKL_NUM_THREADS: $MKL_NUM_THREADS"
echo "  NUMEXPR_NUM_THREADS: $NUMEXPR_NUM_THREADS"

# Verificar memoria disponible
MEM_AVAILABLE=$(grep MemAvailable /proc/meminfo | awk '{print $2}')
echo "Memoria disponible: $((MEM_AVAILABLE / 1024)) MB"

# Ajustar parámetros según memoria
if [ $MEM_AVAILABLE -lt 4000000 ]; then
    echo "Modo: LOWVRAM (menos de 4GB RAM)"
    EXTRA_ARGS="--lowvram --disable-smart-memory"
elif [ $MEM_AVAILABLE -lt 8000000 ]; then
    echo "Modo: NORMALVRAM (menos de 8GB RAM)"
    EXTRA_ARGS="--normalvram"
else
    echo "Modo: HIGHVRAM"
    EXTRA_ARGS=""
fi

# Ejecutar ComfyUI
cd /ComfyUI
exec python main.py \
    --listen 0.0.0.0 \
    --port 8188 \
    --cpu \
    --disable-xformers \
    --disable-cuda-malloc \
    $EXTRA_ARGS