from sys import exit
from PIL import Image
from os import getenv
from redis import Redis
from hashlib import md5
from pathlib import Path
from threading import Lock
from datetime import datetime
from dotenv import load_dotenv
from functools import lru_cache
from psutil import virtual_memory
from json import dumps, load, dump
from typing import Dict, Optional, List, Any
from diffusers import StableDiffusion3Pipeline, DPMSolverMultistepScheduler
from torch import cuda, backends, float16, float32, Generator, inference_mode, autocast

# Cargar variables de entorno
load_dotenv()

class RedisImageCache:
    """Gestor avanzado de cache con Redis para SD3"""
    
    def __init__(self, host='redis', port=6379, db=0):
        self.host = host
        self.port = port
        self.db = db
        self.client = None
        self._connect()
    
    def _connect(self):
        """Establece conexión con Redis"""
        try:
            self.client = Redis(
                host=self.host,
                port=self.port,
                db=self.db,
                decode_responses=False,  # Para datos binarios
                socket_connect_timeout=5,
                socket_keepalive=True,
                retry_on_timeout=True
            )
            self.client.ping()
            print(f"✅ Redis conectado: {self.host}:{self.port}")
            return True
        except Exception as e:
            print(f"❌ Redis no disponible: {e}")
            self.client = None
            return False
    
    def is_connected(self):
        """Verifica conexión"""
        if self.client:
            try:
                return self.client.ping()
            except:
                return False
        return False
    
    def cache_image(self, key: str, image: Image.Image, ttl: int = 7200):
        """Cachea imagen en Redis"""
        if not self.is_connected():
            return False
        
        try:
            # Convertir imagen a bytes optimizados
            from io import BytesIO
            buffer = BytesIO()
            image.save(buffer, format='WEBP', quality=85, optimize=True)
            image_bytes = buffer.getvalue()
            
            # Guardar con clave comprimida
            cache_key = f"sd3_img:{md5(key.encode()).hexdigest()}"
            self.client.setex(cache_key, ttl, image_bytes)
            
            # Guardar metadata
            meta_key = f"sd3_meta:{md5(key.encode()).hexdigest()}"
            metadata = {
                "cached_at": datetime.now().isoformat(),
                "key_hash": key,
                "size": len(image_bytes),
                "format": "WEBP"
            }
            self.client.setex(meta_key, ttl, dumps(metadata))
            
            return True
        except Exception as e:
            print(f"⚠️ Error cacheando imagen: {e}")
            return False
    
    def get_cached_image(self, key: str) -> Optional[Image.Image]:
        """Obtiene imagen del cache"""
        if not self.is_connected():
            return None
        
        try:
            cache_key = f"sd3_img:{md5(key.encode()).hexdigest()}"
            image_bytes = self.client.get(cache_key)
            
            if image_bytes:
                from io import BytesIO
                return Image.open(BytesIO(image_bytes))
            return None
        except Exception as e:
            print(f"⚠️ Error obteniendo imagen cache: {e}")
            return None
    
    def get_stats(self) -> Dict:
        """Obtiene estadísticas del cache"""
        if not self.is_connected():
            return {"status": "disconnected"}
        
        try:
            # Contar claves de SD3
            img_keys = self.client.keys("sd3_img:*")
            meta_keys = self.client.keys("sd3_meta:*")
            
            # Calcular tamaño total
            total_size = 0
            for key in img_keys:
                size = self.client.memory_usage(key)
                if size:
                    total_size += size
            
            return {
                "status": "connected",
                "images_cached": len(img_keys),
                "metadata_entries": len(meta_keys),
                "total_size_mb": total_size / (1024 * 1024),
                "used_memory_mb": int(self.client.info('memory')['used_memory']) / (1024 * 1024)
            }
        except Exception as e:
            return {"status": f"error: {str(e)}"}

class SD3GeneratorSingleton:
    """Singleton optimizado para SD3 con Redis"""
    
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
            self.initialized = False
            self._initialize()
    
    def _initialize(self):
        """Inicialización diferida"""
        print("🚀 Inicializando SD3 Generator con Redis...")
        
        # Configurar directorios
        self.output_dir = Path(getenv('OUTPUT_DIR', '/app/outputs'))
        self.output_dir.mkdir(exist_ok=True)
        
        # Configurar Redis
        redis_host = getenv('REDIS_HOST', 'redis')
        redis_port = int(getenv('REDIS_PORT', 6379))
        self.redis_cache = RedisImageCache(host=redis_host, port=redis_port)
        
        # Configurar dispositivo
        self.device = self._select_device()
        print(f"📱 Dispositivo seleccionado: {self.device}")
        
        # Configurar optimizaciones PyTorch
        self._configure_torch()
        
        # Prompts predefinidos
        self.prompts = self._load_prompts()
        
        # Cache local
        self.local_cache = {}
        
        # Model loading (diferido)
        self.pipe = None
        self.model_loaded = False
        
        self.initialized = True
        print("✅ SD3 Generator inicializado")
    
    def _select_device(self):
        """Selecciona el mejor dispositivo disponible"""
        if cuda.is_available():
            # Verificar memoria GPU
            gpu_memory = cuda.get_device_properties(0).total_memory / (1024**3)
            if gpu_memory >= 6:  # Mínimo 6GB para SD3
                print(f"✅ GPU disponible: {gpu_memory:.1f}GB")
                return "cuda"
        
        # Verificar RAM del sistema
        system_memory = virtual_memory().total / (1024**3)
        if system_memory >= 12:  # Mínimo 12GB para SD3 en CPU
            print(f"⚠️  Usando CPU: {system_memory:.1f}GB RAM")
            return "cpu"
        
        print("❌ Sistema no cumple requisitos mínimos")
        return "cpu"  # Fallback
    
    def _configure_torch(self):
        """Configura optimizaciones de PyTorch"""
        # Optimizaciones generales
        os.environ['TOKENIZERS_PARALLELISM'] = 'false'
        os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
        
        if self.device == "cuda":
            backends.cudnn.benchmark = True
            backends.cuda.matmul.allow_tf32 = True
            backends.cudnn.allow_tf32 = True
    
    def _load_prompts(self) -> Dict[str, Dict]:
        """Carga prompts optimizados para SD3"""
        return {
            "hyperrealistic": {
                "prompt": "hyperrealistic portrait photography of a unique synthetic human, detailed facial features, natural skin texture, professional studio lighting, sharp focus, 8k resolution, cinematic",
                "negative": "blurry, deformed, cartoon, anime, 3d, render, worst quality",
                "steps": 25,
                "resolution": 768
            },
            "professional": {
                "prompt": "corporate headshot of a professional adult, business attire, clean background, professional photography, sharp focus, realistic skin texture",
                "negative": "casual, cartoon, blurry, deformed",
                "steps": 20,
                "resolution": 768
            },
            "cinematic": {
                "prompt": "cinematic portrait of a character, dramatic lighting, film noir style, detailed facial expression, moody atmosphere, photorealistic",
                "negative": "bright, happy, cartoon, anime",
                "steps": 25,
                "resolution": 768
            },
            "scifi": {
                "prompt": "advanced synthetic human with cybernetic features mixed with organic tissue, sci-fi aesthetic, neon lighting, hyperdetailed, futuristic",
                "negative": "historical, primitive, old, blurry",
                "steps": 30,
                "resolution": 768
            }
        }
    
    @lru_cache(maxsize=1)
    def _load_model(self):
        """Carga el modelo SD3 (cached singleton)"""
        print("🔧 Cargando SD3 Medium...")
        
        # Verificar token
        hf_token = getenv('HF_TOKEN')
        if not hf_token:
            print("❌ HF_TOKEN no configurado")
            print("   Configura HF_TOKEN en .env o variables de entorno")
            return None
        
        try:
            # Configurar dtype
            torch_dtype = float16 if self.device == "cuda" else float32
            
            # Cargar modelo
            pipe = StableDiffusion3Pipeline.from_pretrained(
                "stabilityai/stable-diffusion-3-medium-diffusers",
                torch_dtype=torch_dtype,
                token=hf_token,
                use_safetensors=True,
                safety_checker=None,
                cache_dir=getenv('MODEL_CACHE_DIR', '/app/models')
            )
            
            # Optimizaciones
            if self.device == "cpu":
                pipe.enable_sequential_cpu_offload()
                pipe.enable_attention_slicing(1)
            else:
                pipe = pipe.to(self.device)
                try:
                    pipe.enable_xformers_memory_efficient_attention()
                except:
                    pass
                pipe.enable_vae_slicing()
            
            # Scheduler optimizado
            pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True
            )
            
            print("✅ SD3 Medium cargado exitosamente")
            return pipe
            
        except Exception as e:
            print(f"❌ Error cargando SD3: {str(e)[:150]}")
            return None
    
    def get_pipe(self):
        """Obtiene el pipeline (carga lazy si es necesario)"""
        if self.pipe is None:
            self.pipe = self._load_model()
            self.model_loaded = self.pipe is not None
        return self.pipe
    
    def _generate_cache_key(self, prompt: str, negative: str, 
                          resolution: int, steps: int) -> str:
        """Genera clave única para cache"""
        key_data = f"sd3_v1:{prompt}:{negative}:{resolution}:{steps}"
        return hashlib.sha256(key_data.encode()).hexdigest()
    
    def generate_image(self, 
                      prompt: str, 
                      negative_prompt: str = "",
                      resolution: int = 768,
                      steps: int = 25,
                      preset: Optional[str] = None,
                      use_cache: bool = True) -> Optional[Image.Image]:
        """Genera imagen con cache multi-nivel"""
        
        # Usar preset si está especificado
        if preset and preset in self.prompts:
            config = self.prompts[preset]
            prompt = config["prompt"]
            negative_prompt = config["negative"] if not negative_prompt else negative_prompt
            steps = config["steps"]
            resolution = config["resolution"]
        
        # Clave de cache
        cache_key = self._generate_cache_key(prompt, negative_prompt, resolution, steps)
        
        # 1. Cache Redis
        if use_cache and self.redis_cache.is_connected():
            cached = self.redis_cache.get_cached_image(cache_key)
            if cached:
                print("🔄 Imagen obtenida de Redis cache")
                self.local_cache[cache_key] = cached.copy()
                return cached
        
        # 2. Cache local
        if use_cache and cache_key in self.local_cache:
            print("🔄 Imagen obtenida de cache local")
            return self.local_cache[cache_key].copy()
        
        # 3. Generar nueva
        print(f"🎨 Generando nueva imagen {resolution}x{resolution}...")
        
        pipe = self.get_pipe()
        if pipe is None:
            return None
        
        try:
            # Seed determinística
            seed = int(md5(prompt.encode()).hexdigest()[:8], 16) % 2**32
            generator = Generator(device=self.device).manual_seed(seed)
            
            # Inference optimizado
            with inference_mode(), autocast(self.device):
                result = pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=resolution,
                    width=resolution,
                    num_inference_steps=steps,
                    guidance_scale=7.0,
                    generator=generator,
                    num_images_per_prompt=1,
                    output_type="pil"
                )
            
            if result.images:
                image = result.images[0]
                
                # Guardar en caches
                self.local_cache[cache_key] = image.copy()
                
                if self.redis_cache.is_connected():
                    self.redis_cache.cache_image(cache_key, image)
                
                print("✅ Imagen generada y cacheada")
                return image
            
            return None
            
        except cuda.OutOfMemoryError:
            print("⚠️  Memoria GPU insuficiente, reduciendo resolución...")
            return self.generate_image(prompt, negative_prompt, 512, steps, preset, use_cache)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print("⚠️  Memoria insuficiente, usando CPU fallback...")
                return self.generate_image(prompt, negative_prompt, 512, steps, preset, use_cache)
            print(f"❌ Error: {str(e)[:80]}")
            return None
    
    def save_result(self, image: Image.Image, prompt: str, 
                   negative: str, style: str) -> Dict[str, Any]:
        """Guarda resultado con metadata"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Nombre de archivo
        safe_style = "".join(c for c in style if c.isalnum() or c in ('_', '-')).rstrip()
        filename = f"sd3_{safe_style}_{timestamp}.webp"
        filepath = self.output_dir / filename
        
        # Guardar imagen (WebP para mejor compresión)
        image.save(filepath, 'WEBP', quality=90, optimize=True)
        
        # Metadata
        metadata = {
            "id": md5(f"{prompt}{timestamp}".encode()).hexdigest()[:16],
            "timestamp": timestamp,
            "prompt": prompt,
            "negative_prompt": negative,
            "style": style,
            "resolution": f"{image.width}x{image.height}",
            "model": "Stable Diffusion 3 Medium",
            "device": self.device,
            "redis_cached": self.redis_cache.is_connected(),
            "filepath": str(filepath)
        }
        
        # Guardar metadata
        meta_file = self.output_dir / f"sd3_meta_{timestamp}.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Guardado: {filename}")
        return metadata
    
    def get_system_info(self) -> Dict[str, Any]:
        """Obtiene información del sistema"""
        info = {
            "device": self.device,
            "model_loaded": self.model_loaded,
            "redis": self.redis_cache.get_stats(),
            "local_cache_size": len(self.local_cache)
        }
        
        # Información de memoria
        try:
            mem = virtual_memory()
            info["memory"] = {
                "total_gb": mem.total / (1024**3),
                "available_gb": mem.available / (1024**3),
                "percent_used": mem.percent
            }
            
            if self.device == "cuda":
                info["gpu_memory_gb"] = cuda.get_device_properties(0).total_memory / (1024**3)
        except:
            pass
        
        return info

# Interfaz de usuario mejorada
def interactive_mode():
    """Modo interactivo con Redis"""
    print("\n" + "="*80)
    print("🎬 SD3 MEDIUM GENERATOR - INTERACTIVE MODE")
    print("="*80)
    
    generator = SD3GeneratorSingleton()
    info = generator.get_system_info()
    
    print("\n📊 SYSTEM INFO:")
    print(f"   • Device: {info['device']}")
    print(f"   • Model: {'✅ Loaded' if info['model_loaded'] else '❌ Not loaded'}")
    print(f"   • Redis: {info['redis'].get('status', 'unknown')}")
    
    if info['redis'].get('status') == 'connected':
        print(f"   • Images in cache: {info['redis'].get('images_cached', 0)}")
    
    # Mostrar presets
    print("\n🎭 AVAILABLE PRESETS:")
    for i, (key, config) in enumerate(generator.prompts.items(), 1):
        print(f"   {i}. {key.title()} - {config['resolution']}px, {config['steps']} steps")
    
    print("   5. Custom prompt")
    
    choice = input("\nSelect preset (1-5): ").strip()
    
    if choice == "5":
        # Personalizado
        print("\n✏️ CUSTOM PROMPT:")
        prompt = input("Enter prompt: ").strip()
        if not prompt:
            prompt = "hyperrealistic portrait of a person"
        
        negative = input("Negative prompt (optional): ").strip()
        if not negative:
            negative = "blurry, deformed, cartoon, anime"
        
        style = "custom"
        
        print("\n📐 RESOLUTION:")
        print("   1. 512x512 (Fast)")
        print("   2. 768x768 (Recommended)")
        print("   3. 1024x1024 (High quality)")
        
        res_choice = input("Select (1-3): ").strip()
        resolutions = {"1": 512, "2": 768, "3": 1024}
        resolution = resolutions.get(res_choice, 768)
        
        steps = 25 if resolution <= 768 else 30
        
    else:
        # Usar preset
        presets = list(generator.prompts.keys())
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(presets):
                preset = presets[idx]
                config = generator.prompts[preset]
                prompt = config["prompt"]
                negative = config["negative"]
                resolution = config["resolution"]
                steps = config["steps"]
                style = preset
            else:
                print("⚠️ Invalid selection, using default")
                prompt, negative, resolution, steps, style = generator.prompts["hyperrealistic"].values()
                style = "hyperrealistic"
        except:
            print("⚠️ Invalid selection, using default")
            prompt, negative, resolution, steps, style = generator.prompts["hyperrealistic"].values()
            style = "hyperrealistic"
    
    # Generar
    print(f"\n🚀 Generating {resolution}x{resolution} image...")
    
    image = generator.generate_image(
        prompt=prompt,
        negative_prompt=negative,
        resolution=resolution,
        steps=steps,
        preset=None if choice == "5" else style
    )
    
    if image:
        metadata = generator.save_result(image, prompt, negative, style)
        
        print("\n" + "="*80)
        print("✅ GENERATION COMPLETE")
        print("="*80)
        
        print(f"\n📁 FILE: {Path(metadata['filepath']).name}")
        print(f"📐 RESOLUTION: {metadata['resolution']}")
        print(f"🎭 STYLE: {metadata['style']}")
        print(f"📂 LOCATION: {metadata['filepath']}")
        
        print(f"\n🔧 CONFIGURATION:")
        print(f"   • Model: {metadata['model']}")
        print(f"   • Device: {metadata['device']}")
        print(f"   • Redis cached: {metadata['redis_cached']}")
        
        return metadata
    else:
        print("\n❌ Generation failed")
        return None

def batch_mode():
    """Genera múltiples imágenes"""
    print("\n🔢 BATCH MODE")
    
    generator = SD3GeneratorSingleton()
    
    print("\nAvailable presets:")
    presets = list(generator.prompts.keys())
    for i, preset in enumerate(presets, 1):
        print(f"   {i}. {preset}")
    
    selection = input("\nSelect presets (e.g., '1,2,3' or 'all'): ").strip().lower()
    
    if selection == 'all':
        selected_presets = presets
    else:
        try:
            indices = [int(x.strip()) - 1 for x in selection.split(',')]
            selected_presets = [presets[i] for i in indices if 0 <= i < len(presets)]
        except:
            print("❌ Invalid selection, using all")
            selected_presets = presets[:2]  # Limitar a 2 por defecto
    
    print(f"\n🔄 Generating {len(selected_presets)} images...")
    
    results = []
    for preset in selected_presets:
        print(f"\n🎭 Generating {preset}...")
        
        image = generator.generate_image(preset=preset)
        if image:
            config = generator.prompts[preset]
            metadata = generator.save_result(image, config["prompt"], config["negative"], preset)
            results.append(metadata)
            print(f"   ✅ {Path(metadata['filepath']).name}")
        else:
            print(f"   ❌ Failed")
    
    print(f"\n📊 Batch complete: {len(results)}/{len(selected_presets)} successful")
    return results

def quick_mode():
    """Modo rápido"""
    print("\n⚡ QUICK MODE")
    
    generator = SD3GeneratorSingleton()
    
    # Usar preset hyperrealistic
    image = generator.generate_image(preset="hyperrealistic")
    
    if image:
        config = generator.prompts["hyperrealistic"]
        metadata = generator.save_result(image, config["prompt"], config["negative"], "quick")
        
        print(f"\n✅ Generated: {Path(metadata['filepath']).name}")
        print(f"📍 Location: {metadata['filepath']}")
        return metadata
    else:
        print("❌ Generation failed")
        return None

def main():
    """Función principal"""
    print("="*80)
    print("🚀 STABLE DIFFUSION 3 MEDIUM + REDIS OPTIMIZED")
    print("   Docker Compose | Multi-level Cache | High Performance")
    print("="*80)
    
    # Verificar token
    if not getenv('HF_TOKEN'):
        print("\n❌ HF_TOKEN not configured!")
        print("   Set HF_TOKEN in .env file or environment variables")
        print("   Get token from: https://huggingface.co/settings/tokens")
        return
    
    # Menú principal
    print("\n🎯 EXECUTION MODES:")
    print("   1. Interactive mode (choose prompts)")
    print("   2. Quick mode (generate now)")
    print("   3. Batch mode (multiple images)")
    print("   4. System info")
    
    mode = input("\nSelect mode (1-4): ").strip() or "1"
    
    if mode == "1":
        interactive_mode()
    elif mode == "2":
        quick_mode()
    elif mode == "3":
        batch_mode()
    elif mode == "4":
        generator = SD3GeneratorSingleton()
        info = generator.get_system_info()
        print("\n📊 SYSTEM INFORMATION:")
        print(dumps(info, indent=2, default=str))
    
    print("\n" + "="*80)
    print("✨ Process completed")

if __name__ == "__main__":
    # Configuración para Docker
    import signal
    
    def handle_exit(signum, frame):
        print("\n\n👋 Shutting down gracefully...")
        exit(0)
    
    signal.signal(signal.SIGINT, handle_exit)
    signal.signal(signal.SIGTERM, handle_exit)
    
    # Ejecutar
    main()