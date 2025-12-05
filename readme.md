sudo dnf install -y python3-pip python3-devel cmake make gcc-c++

python -m venv venv
source venv/bin/activate

docker run -it --rm -e HF_TOKEN=tu_token_aqui -v $(pwd)/outputs:/app/outputs -v $(pwd)/models:/app/models prueba_stablediffusion3pipeline-sd3-app python src/main.py 