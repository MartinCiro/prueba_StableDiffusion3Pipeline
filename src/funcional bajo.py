from sys import argv
from PIL import Image
from hashlib import md5
from redis import Redis
from pathlib import Path
from threading import Lock
from datetime import datetime
from os import environ, getenv
from functools import lru_cache
from signal import signal, SIGINT
from typing import Optional, Dict
from torch import cuda, float32, Generator, inference_mode
from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler

# Cache con Redis
class RedisImageCache:
    def __init__(self, host='localhost', port=6379, db=0):
        self.host = host
        self.port = port
        self.db = db
        self.client = None
        self._connect()
    
    def _connect(self):
        try:
            self.client = Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False,
                socket_connect_timeout=3,
                socket_keepalive=True
            )
            self.client.ping()
            print(f"✅ Redis conectado: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Redis no disponible: {e}")
            self.client = None
            return False
    
    def cache_image(self, key: str, image: Image.Image, ttl: int = 3600):
        if not self.client:
            return False
        
        try:
            from io import BytesIO
            buffer = BytesIO()
            image.save(buffer, format='WEBP', quality=85, optimize=True)
            self.client.setex(f"img:{key}", ttl, buffer.getvalue())
            return True
        except:
            return False
    
    def get_cached_image(self, key: str):
        if not self.client:
            return None
        
        try:
            image_bytes = self.client.get(f"img:{key}")
            if image_bytes:
                from io import BytesIO
                return Image.open(BytesIO(image_bytes))
        except:
            pass
        return None

class FullBodyGenerator:
    """Generador especializado en cuerpos completos"""
    
    _instance = None
    _lock = Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not hasattr(self, 'initialized'):
            print("🤖 Iniciando Generador de Cuerpo Completo...")
            
            # Configurar para mínimo consumo
            environ['TOKENIZERS_PARALLELISM'] = 'false'
            environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:16'
            
            # Forzar CPU si no hay suficiente memoria GPU
            if cuda.is_available():
                gpu_memory = cuda.get_device_properties(0).total_memory / (1024**3)
                self.device = "cuda" if gpu_memory >= 4 else "cpu"
            else:
                self.device = "cpu"
            
            print(f"📱 Dispositivo: {self.device.upper()}")
            
            # Redis cache
            redis_host = getenv('REDIS_HOST', 'localhost')
            redis_port = int(getenv('REDIS_PORT', 6379))
            self.redis_cache = RedisImageCache(host=redis_host, port=redis_port)
            
            # Directorios
            self.output_dir = Path(getenv('OUTPUT_DIR', './fullbody_outputs'))
            self.output_dir.mkdir(exist_ok=True)
            
            # Modelo liviano
            self.model_id = "runwayml/stable-diffusion-v1-5"
            
            # Cache local
            self.local_cache = {}
            self.pipe = None
            self.model_loaded = False
            
            # PROMPTS ESPECIALIZADOS EN CUERPO COMPLETO
            self.prompts = {
                "fullbody_portrait": {
                    "prompt": "full body portrait of a person standing, full body visible from head to toes, natural pose, clean background, photorealistic, detailed clothing",
                    "negative": "cropped, cut off, partial body, only face, close-up, blurry, deformed, missing limbs",
                    "steps": 15,
                    "width": 384,   # Más ancho para cuerpo completo
                    "height": 512   # Más alto para mostrar pies
                },
                "fullbody_casual": {
                    "prompt": "full body shot of a person in casual clothes, standing naturally, full figure visible, street photography style, full length",
                    "negative": "closeup, portrait, cropped, sitting, lying down, partial view",
                    "steps": 12,
                    "width": 384,
                    "height": 512
                },
                "fullbody_formal": {
                    "prompt": "full body professional photo of person in formal attire, standing pose, entire body visible, studio lighting, sharp focus",
                    "negative": "casual, close-up, cropped, informal, blurry, distorted proportions",
                    "steps": 15,
                    "width": 384,
                    "height": 512
                },
                "fullbody_active": {
                    "prompt": "full body action shot of person in motion, athletic pose, sports clothing, dynamic composition, entire body visible",
                    "negative": "static, sitting, lying, cropped, blurry motion",
                    "steps": 18,
                    "width": 384,
                    "height": 512
                },
                "fullbody_simple": {
                    "prompt": "person standing, full body visible, plain background, simple pose",
                    "negative": "closeup, cropped, complex background, multiple people",
                    "steps": 10,
                    "width": 384,
                    "height": 512
                }
            }
            
            self.initialized = True
            print("✅ Generador de cuerpo completo inicializado")
    
    @lru_cache(maxsize=1)
    def _load_model(self):
        """Carga el modelo de forma lazy"""
        print("🔧 Cargando modelo...")
        
        try:
            # Usar float32 para CPU
            dtype = float32
            
            pipe = StableDiffusionPipeline.from_pretrained(
                self.model_id,
                torch_dtype=dtype,
                safety_checker=None,
                requires_safety_checker=False
            )
            
            # Optimizaciones mínimas
            pipe.to(self.device)
            pipe.enable_attention_slicing(1)
            
            # Scheduler rápido
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++"
            )
            
            print(f"✅ Modelo cargado: {self.model_id}")
            return pipe
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {e}")
            return None
    
    def get_pipe(self):
        if self.pipe is None:
            self.pipe = self._load_model()
            self.model_loaded = self.pipe is not None
        return self.pipe
    
    def generate_fullbody(self, 
                         prompt: str = None,
                         negative_prompt: str = "",
                         width: int = 384,
                         height: int = 512,
                         steps: int = 15,
                         preset: str = None,
                         use_cache: bool = True) -> Optional[Image.Image]:
        """Genera imagen de cuerpo completo"""
        
        # Usar preset si se especifica
        if preset and preset in self.prompts:
            config = self.prompts[preset]
            prompt = config["prompt"]
            negative_prompt = config["negative"]
            steps = config["steps"]
            width = config["width"]
            height = config["height"]
        
        # Si no hay prompt, usar default
        if not prompt:
            prompt = "full body portrait of a person standing, entire body visible"
        
        # Asegurar relación de aspecto para cuerpo completo (vertical)
        if height < width:
            height, width = width, height  # Hacer vertical si es horizontal
        
        # Limitar tamaño para bajo consumo
        max_size = 512
        if width > max_size or height > max_size:
            scale = max_size / max(width, height)
            width = int(width * scale)
            height = int(height * scale)
            print(f"⚠️  Tamaño ajustado a {width}x{height} para bajo consumo")
        
        # Generar clave de cache
        cache_key = md5(
            f"fullbody:{prompt}:{negative_prompt}:{width}x{height}:{steps}".encode()
        ).hexdigest()
        
        # 1. Cache Redis
        if use_cache and self.redis_cache.client:
            cached = self.redis_cache.get_cached_image(cache_key)
            if cached:
                print("🔄 Imagen de Redis cache")
                self.local_cache[cache_key] = cached.copy()
                return cached
        
        # 2. Cache local
        if use_cache and cache_key in self.local_cache:
            print("🔄 Imagen de cache local")
            return self.local_cache[cache_key].copy()
        
        # 3. Generar nueva
        pipe = self.get_pipe()
        if not pipe:
            return None
        
        print(f"🎨 Generando cuerpo completo {width}x{height}...")
        print(f"📝 Prompt: {prompt[:80]}...")
        
        try:
            # Seed determinística
            seed = hash(prompt) % 2**32
            generator = Generator(device=self.device).manual_seed(seed)
            
            # Inference
            with inference_mode():
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=height,  # Alto para cuerpo completo
                    width=width,    # Ancho adecuado
                    num_inference_steps=steps,
                    guidance_scale=7.0,
                    generator=generator,
                    num_images_per_prompt=1,
                    output_type="pil"
                )
            
            if result.images:
                image = result.images[0]
                
                # Verificar que sea cuerpo completo (al menos 2:3 de relación)
                if image.height / image.width < 1.3:
                    print("⚠️  Posiblemente no es cuerpo completo, regenerando...")
                    # Añadir keywords para cuerpo completo
                    prompt += ", full body, entire figure visible"
                    return self.generate_fullbody(prompt, negative_prompt, width, height, steps, preset, False)
                
                # Guardar en caches
                self.local_cache[cache_key] = image.copy()
                if self.redis_cache.client:
                    self.redis_cache.cache_image(cache_key, image)
                
                print("✅ Cuerpo completo generado")
                return image
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                print("⚠️  Memoria insuficiente, reduciendo tamaño...")
                return self.generate_fullbody(prompt, negative_prompt, 
                                            int(width * 0.8), int(height * 0.8), 
                                            steps - 2, preset, use_cache)
            print(f"❌ Error: {e}")
        
        return None
    
    def quick_generate(self, preset: str = "fullbody_portrait") -> Optional[Dict]:
        """Generación rápida de cuerpo completo"""
        print(f"⚡ Generando {preset}...")
        
        image = self.generate_fullbody(preset=preset)
        
        if image:
            # Guardar
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fullbody_{preset}_{timestamp}.jpg"
            filepath = self.output_dir / filename
            
            image.save(filepath, 'JPEG', quality=90, optimize=True)
            
            metadata = {
                "file": filename,
                "preset": preset,
                "resolution": f"{image.width}x{image.height}",
                "aspect_ratio": f"{image.width/image.height:.2f}:1",
                "device": self.device,
                "timestamp": timestamp,
                "type": "full_body"
            }
            
            print(f"💾 Guardado: {filename}")
            print(f"📐 Tamaño: {image.width}x{image.height}")
            
            return metadata
        
        print("❌ Generación falló")
        return None
    
    def generate_custom_fullbody(self, 
                               clothing: str = "casual clothes",
                               pose: str = "standing naturally",
                               background: str = "clean background",
                               gender: str = "person") -> Optional[Dict]:
        """Genera cuerpo completo con descripción personalizada"""
        
        prompt = f"full body portrait of a {gender} in {clothing}, {pose}, {background}, entire body visible from head to toes, photorealistic"
        
        negative = "cropped, cut off, partial body, close-up, blurry, deformed, missing limbs"
        
        print(f"🎨 Generando cuerpo completo personalizado...")
        print(f"👕 Ropa: {clothing}")
        print(f"🧍 Pose: {pose}")
        
        image = self.generate_fullbody(
            prompt=prompt,
            negative_prompt=negative,
            width=384,
            height=512,
            steps=15
        )
        
        if image:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"fullbody_custom_{timestamp}.jpg"
            filepath = self.output_dir / filename
            
            image.save(filepath, 'JPEG', quality=90, optimize=True)
            
            metadata = {
                "file": filename,
                "clothing": clothing,
                "pose": pose,
                "background": background,
                "gender": gender,
                "resolution": f"{image.width}x{image.height}",
                "timestamp": timestamp
            }
            
            print(f"✅ Guardado: {filename}")
            return metadata
        
        return None
    
    def system_info(self) -> Dict:
        """Información del sistema"""
        info = {
            "device": self.device,
            "model_loaded": self.model_loaded,
            "model": self.model_id,
            "output_dir": str(self.output_dir),
            "redis_connected": self.redis_cache.client is not None,
            "presets_available": list(self.prompts.keys()),
            "fullbody_specialized": True
        }
        
        return info

# Interfaz mejorada para cuerpo completo
def fullbody_menu():
    """Menú para generación de cuerpo completo"""
    print("\n" + "="*60)
    print("🧍 GENERADOR DE CUERPO COMPLETO")
    print("="*60)
    
    generator = FullBodyGenerator()
    info = generator.system_info()
    
    print(f"\n📊 Sistema:")
    print(f"   • Dispositivo: {info['device'].upper()}")
    print(f"   • Modelo: {'✅ Cargado' if info['model_loaded'] else '❌ No cargado'}")
    print(f"   • Redis: {'✅ Conectado' if info['redis_connected'] else '❌ No disponible'}")
    
    print("\n🎭 TIPOS DE CUERPO COMPLETO:")
    presets = list(generator.prompts.keys())
    for i, preset in enumerate(presets, 1):
        config = generator.prompts[preset]
        print(f"   {i}. {preset.replace('_', ' ').title()}")
        print(f"      📐 {config['width']}x{config['height']} | ⏱️  {config['steps']} pasos")
    
    print(f"   {len(presets)+1}. Personalizado")
    print(f"   {len(presets)+2}. Salir")
    
    try:
        choice = input("\nSelecciona (1-{}): ".format(len(presets)+2)).strip()
        
        if choice == str(len(presets)+2):
            print("👋 Adiós!")
            return
        
        if choice == str(len(presets)+1):
            # Modo personalizado
            print("\n🎨 PERSONALIZAR CUERPO COMPLETO:")
            
            print("\n👕 Tipo de ropa:")
            print("   1. Casual")
            print("   2. Formal")
            print("   3. Deportiva")
            print("   4. Elegante")
            clothing_choice = input("   Selecciona (1-4): ").strip()
            clothing_options = ["casual clothes", "formal attire", "sports clothing", "elegant outfit"]
            clothing = clothing_options[int(clothing_choice)-1] if clothing_choice in ['1','2','3','4'] else "casual clothes"
            
            print("\n🧍 Pose:")
            print("   1. De pie natural")
            print("   2. De pie formal")
            print("   3. En movimiento")
            print("   4. Relajado")
            pose_choice = input("   Selecciona (1-4): ").strip()
            pose_options = ["standing naturally", "standing formally", "in motion", "relaxed stance"]
            pose = pose_options[int(pose_choice)-1] if pose_choice in ['1','2','3','4'] else "standing naturally"
            
            print("\n🏞️ Fondo:")
            print("   1. Fondo limpio")
            print("   2. Estudio")
            print("   3. Exterior")
            print("   4. Abstracto")
            bg_choice = input("   Selecciona (1-4): ").strip()
            bg_options = ["clean background", "studio background", "outdoor setting", "abstract background"]
            background = bg_options[int(bg_choice)-1] if bg_choice in ['1','2','3','4'] else "clean background"
            
            print("\n👤 Persona:")
            print("   1. Persona (neutral)")
            print("   2. Hombre")
            print("   3. Mujer")
            gender_choice = input("   Selecciona (1-3): ").strip()
            gender_options = ["person", "man", "woman"]
            gender = gender_options[int(gender_choice)-1] if gender_choice in ['1','2','3'] else "person"
            
            result = generator.generate_custom_fullbody(clothing, pose, background, gender)
            
        else:
            # Preset normal
            idx = int(choice) - 1
            if 0 <= idx < len(presets):
                preset = presets[idx]
                result = generator.quick_generate(preset)
            else:
                print("⚠️  Opción inválida")
                return
        
        if result:
            print(f"\n✅ GENERACIÓN COMPLETA")
            print(f"📁 Archivo: {result['file']}")
            print(f"📐 Resolución: {result['resolution']}")
            print(f"📂 Carpeta: {generator.output_dir}")
            
            # Mostrar detalles si es personalizado
            if 'clothing' in result:
                print(f"👕 Ropa: {result['clothing']}")
                print(f"🧍 Pose: {result['pose']}")
                print(f"🏞️ Fondo: {result['background']}")
    
    except (ValueError, KeyboardInterrupt):
        print("\n⚠️  Operación cancelada")

def quick_fullbody():
    """Ejecución automática rápida"""
    print("🚀 Generando cuerpo completo automático...")
    
    generator = FullBodyGenerator()
    
    if not generator.model_loaded:
        print("❌ Modelo no disponible")
        return
    
    # Generar el preset más común
    result = generator.quick_generate("fullbody_portrait")
    
    if result:
        print(f"\n✅ Éxito: {result['file']}")
        print(f"📍 En: {generator.output_dir}")
        return result
    else:
        print("❌ Falló")
        return None

def main():
    """Punto de entrada principal"""
    
    # Manejar signals
    signal(SIGINT, lambda s, f: print("\n\n⚠️  Interrumpido por usuario"))
    
    # Modo simple: si hay argumento, modo automático
    if len(argv) > 1:
        if argv[1] == "auto":
            quick_fullbody()
        elif argv[1] == "info":
            generator = FullBodyGenerator()
            import json
            print(json.dumps(generator.system_info(), indent=2))
        elif argv[1] == "custom":
            generator = FullBodyGenerator()
            result = generator.generate_custom_fullbody()
            if result:
                print(f"✅ Generado: {result['file']}")
        else:
            print("Uso: python script.py [auto|info|custom]")
    else:
        # Modo interactivo
        fullbody_menu()
    
    print("\n✨ Proceso completado")

if __name__ == "__main__":
    main()