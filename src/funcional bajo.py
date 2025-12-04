from diffusers import StableDiffusionPipeline, DPMSolverMultistepScheduler
import torch
from pathlib import Path
import hashlib
from typing import List, Optional, Dict, Any
import concurrent.futures
from threading import Lock
from PIL import Image
from datetime import datetime
import json
import os

class OptimizedHumanGenerator:
    """Generador de humanos optimizado con singleton pattern y cache"""
    
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
            print("🤖 Generador Optimizado Cargando...")
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            self.output_dir = Path("optimized_humans")
            self.output_dir.mkdir(exist_ok=True)
            
            # Configurar para mínimo consumo
            os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:32'
            os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
            
            print(f"📱 Dispositivo: {self.device}")
            self._load_optimized_model()
            
            # Cache de prompts predefinidos
            self.preset_prompts = self._load_preset_prompts()
            
            # Cache de resultados
            self._result_cache = {}
            
            self.initialized = True
            print("✅ Generador listo")
    
    def _load_optimized_model(self):
        """Carga el modelo con optimizaciones específicas"""
        print("🔧 Cargando modelo optimizado...")
        
        MODEL_ID = "stabilityai/stable-diffusion-2-1"
        print(f"   Modelo: {MODEL_ID}")
        
        # Usar precisión mixta si hay GPU
        torch_dtype = torch.float16 if self.device == "cuda" else torch.float32
        print(f"   Precisión: {torch_dtype}")
        
        try:
            self.pipe = StableDiffusionPipeline.from_pretrained(
                MODEL_ID,
                torch_dtype=torch_dtype,
                safety_checker=None,
                requires_safety_checker=False,
                use_safetensors=True
            )
            
            # Optimizaciones específicas por dispositivo
            if self.device == "cuda":
                try:
                    self.pipe.enable_xformers_memory_efficient_attention()
                    print("   ✅ XFormers activado")
                except:
                    print("   ⚠️  XFormers no disponible")
                
                torch.backends.cudnn.benchmark = True
            
            # Atención slicing para todos
            self.pipe.enable_attention_slicing()
            
            self.pipe.to(self.device)
            
            # Scheduler más rápido (2x speed)
            self.pipe.scheduler = DPMSolverMultistepScheduler.from_config(
                self.pipe.scheduler.config,
                algorithm_type="dpmsolver++",
                use_karras_sigmas=True
            )
            
            print("✅ Modelo cargado con optimizaciones")
            
        except Exception as e:
            print(f"❌ Error cargando modelo: {str(e)[:100]}")
            # Fallback a modelo más ligero
            self._load_fallback_model()
    
    def _load_fallback_model(self):
        """Carga modelo alternativo si falla el principal"""
        try:
            print("🔄 Cargando modelo fallback...")
            self.pipe = StableDiffusionPipeline.from_pretrained(
                "runwayml/stable-diffusion-v1-5",
                torch_dtype=torch.float32
            )
            self.pipe.to(self.device)
            self.pipe.enable_attention_slicing()
            print("✅ Modelo fallback cargado")
        except Exception as e:
            print(f"❌ Error crítico: {e}")
            self.pipe = None
    
    def _load_preset_prompts(self) -> Dict[str, Dict[str, str]]:
        """Carga prompts predefinidos optimizados"""
        return {
            "realistic_young": {
                "prompt": "professional portrait of young adult, detailed face, studio lighting, photorealistic, 8k",
                "negative": "blurry, deformed, cartoon, anime, 3d, doll",
                "steps": 15
            },
            "professional_adult": {
                "prompt": "corporate headshot of professional adult, sharp focus, realistic, business attire",
                "negative": "child, young, cartoon, painting, sketch",
                "steps": 15
            },
            "diverse_portrait": {
                "prompt": "hyperrealistic portrait of diverse person, cinematic lighting, detailed skin texture",
                "negative": "blurry, bad anatomy, deformed, mutated",
                "steps": 18
            },
            "neutral_face": {
                "prompt": "studio portrait neutral expression, clean background, professional photography",
                "negative": "emotional, cartoon, anime, 3d render",
                "steps": 12
            },
            "futuristic_human": {
                "prompt": "futuristic synthetic human portrait, cybernetic features, neon lighting, sci-fi",
                "negative": "historical, medieval, ancient, primitive",
                "steps": 20
            }
        }
    
    def generate_image(self, 
                      prompt: str, 
                      negative_prompt: str = "",
                      resolution: int = 512,
                      steps: Optional[int] = None,
                      preset: Optional[str] = None) -> Optional[Image.Image]:
        """Genera una imagen optimizada"""
        
        if self.pipe is None:
            print("❌ Modelo no disponible")
            return None
        
        # Usar configuración de preset si se especifica
        if preset and preset in self.preset_prompts:
            preset_config = self.preset_prompts[preset]
            prompt = preset_config["prompt"]
            negative_prompt = preset_config["negative"]
            steps = preset_config["steps"] if steps is None else steps
        
        # Generar hash para cache
        cache_key = f"{prompt}_{negative_prompt}_{resolution}_{steps}"
        
        if cache_key in self._result_cache:
            print("🔄 Usando imagen cacheada")
            return self._result_cache[cache_key].copy()
        
        print(f"🎨 Generando {resolution}x{resolution} (steps: {steps or 15})...")
        print(f"   Prompt: {prompt[:60]}...")
        
        try:
            # Seed determinística basada en prompt
            seed = int(hashlib.md5(prompt.encode()).hexdigest()[:8], 16) % 2**32
            generator = torch.Generator(device=self.device).manual_seed(seed)
            
            # Inference mode para reducir memoria
            with torch.inference_mode():
                result = self.pipe(
                    prompt=prompt,
                    negative_prompt=negative_prompt,
                    height=resolution,
                    width=resolution,
                    num_inference_steps=steps or 15,  # Default más bajo
                    guidance_scale=7.0,  # Reducido para velocidad
                    generator=generator,
                    num_images_per_prompt=1,
                    output_type="pil"
                )
            
            if result.images and len(result.images) > 0:
                image = result.images[0]
                # Cachear resultado
                self._result_cache[cache_key] = image.copy()
                print("✅ Imagen generada")
                return image
            else:
                print("❌ No se generó imagen")
                return None
                
        except torch.cuda.OutOfMemoryError:
            print("⚠️  Memoria insuficiente, reduciendo resolución...")
            return self.generate_image(prompt, negative_prompt, 384, steps, preset)
        except Exception as e:
            print(f"❌ Error: {str(e)[:80]}")
            return None
    
    def generate_batch(self, 
                      prompts: List[str],
                      resolution: int = 512,
                      steps: int = 15,
                      max_workers: int = 2) -> List[Optional[Image.Image]]:
        """Genera múltiples imágenes en paralelo"""
        
        print(f"🔢 Generando batch de {len(prompts)} imágenes...")
        
        images = []
        
        # Usar ThreadPool para paralelizar I/O y preparación
        with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for prompt in prompts:
                # Cada prompt en su propio hilo
                future = executor.submit(
                    self.generate_image,
                    prompt=prompt,
                    resolution=resolution,
                    steps=steps
                )
                futures.append(future)
            
            # Recoger resultados
            for i, future in enumerate(concurrent.futures.as_completed(futures), 1):
                image = future.result()
                images.append(image)
                print(f"   [{i}/{len(prompts)}] {'✅' if image else '❌'}")
        
        return images
    
    def save_image(self, 
                  image: Image.Image,
                  prompt: str,
                  negative_prompt: str = "",
                  preset: str = "custom",
                  quality: int = 90) -> str:
        """Guarda la imagen y metadata"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Nombre de archivo seguro
        safe_preset = "".join(c for c in preset if c.isalnum() or c in ('_', '-')).rstrip()
        filename = f"human_{safe_preset}_{timestamp}.jpg"
        filepath = self.output_dir / filename
        
        # Guardar imagen optimizada
        image.save(
            filepath, 
            'JPEG', 
            quality=quality, 
            optimize=True,
            progressive=True
        )
        
        # Metadata
        metadata = {
            "timestamp": timestamp,
            "prompt": prompt,
            "negative_prompt": negative_prompt,
            "preset": preset,
            "resolution": f"{image.width}x{image.height}",
            "model": "Stable Diffusion 2.1",
            "device": self.device,
            "quality": quality
        }
        
        # Guardar metadata como JSON
        meta_file = self.output_dir / f"meta_{timestamp}.json"
        with open(meta_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2, ensure_ascii=False)
        
        print(f"💾 Guardado: {filename}")
        return str(filepath)
    
    def interactive_mode(self):
        """Modo interactivo simplificado"""
        
        if self.pipe is None:
            print("❌ Modelo no disponible")
            return
        
        print("\n" + "="*50)
        print("🤖 GENERADOR INTERACTIVO")
        print("="*50)
        
        # Seleccionar tipo
        print("\n🎭 TIPOS DISPONIBLES:")
        for i, (key, config) in enumerate(self.preset_prompts.items(), 1):
            print(f"   {i}. {key.replace('_', ' ').title()}")
        print("   6. Personalizado")
        
        choice = input("\nSelecciona (1-6): ").strip()
        
        if choice == "6":
            # Personalizado
            prompt = input("Prompt: ").strip()
            if not prompt:
                prompt = "professional portrait of a person"
            
            negative = input("Negative prompt (opcional): ").strip()
            preset = "custom"
        else:
            # Convertir elección a preset
            choices = list(self.preset_prompts.keys())
            try:
                idx = int(choice) - 1
                if 0 <= idx < len(choices):
                    preset = choices[idx]
                    config = self.preset_prompts[preset]
                    prompt = config["prompt"]
                    negative = config["negative"]
                else:
                    print("⚠️  Opción inválida, usando realista")
                    preset = "realistic_young"
                    config = self.preset_prompts[preset]
                    prompt = config["prompt"]
                    negative = config["negative"]
            except:
                print("⚠️  Opción inválida, usando realista")
                preset = "realistic_young"
                config = self.preset_prompts[preset]
                prompt = config["prompt"]
                negative = config["negative"]
        
        # Resolución
        print("\n📐 RESOLUCIÓN:")
        print("   1. 512x512 (Rápido)")
        print("   2. 768x768 (Calidad)")
        
        res_choice = input("Selecciona (1-2): ").strip()
        resolution = 512 if res_choice == "1" else 768
        
        # Generar
        print(f"\n🚀 Generando...")
        image = self.generate_image(
            prompt=prompt,
            negative_prompt=negative,
            resolution=resolution,
            preset=preset if choice != "6" else None
        )
        
        if image:
            saved_path = self.save_image(image, prompt, negative, preset)
            print(f"\n✅ Generación completa!")
            print(f"📁 Archivo: {Path(saved_path).name}")
            print(f"📂 Ubicación: {saved_path}")
            return saved_path
        else:
            print("\n❌ Generación falló")
            return None
    
    def quick_generate(self, preset: str = "realistic_young") -> Optional[str]:
        """Generación rápida con defaults"""
        
        print(f"⚡ Generación rápida: {preset}")
        
        if preset not in self.preset_prompts:
            preset = "realistic_young"
        
        config = self.preset_prompts[preset]
        
        image = self.generate_image(
            prompt=config["prompt"],
            negative_prompt=config["negative"],
            resolution=512,
            steps=config["steps"],
            preset=preset
        )
        
        if image:
            saved_path = self.save_image(
                image,
                config["prompt"],
                config["negative"],
                preset
            )
            return saved_path
        
        return None
    
    def clear_cache(self):
        """Limpia el cache de imágenes"""
        self._result_cache.clear()
        print("🗑️  Cache limpiado")

# Funciones de utilidad
def quick_start():
    """Inicio rápido para uso inmediato"""
    print("\n⚡ INICIO RÁPIDO")
    generator = OptimizedHumanGenerator()
    
    if generator.pipe is None:
        print("❌ No se pudo cargar el modelo")
        return None
    
    saved_path = generator.quick_generate()
    
    if saved_path:
        print(f"\n✅ Humano generado: {Path(saved_path).name}")
    else:
        print("❌ Generación falló")
    
    return saved_path

def batch_generate(count: int = 3):
    """Genera múltiples imágenes"""
    print(f"\n🔢 GENERANDO {count} IMÁGENES")
    
    generator = OptimizedHumanGenerator()
    
    if generator.pipe is None:
        print("❌ Modelo no disponible")
        return
    
    # Usar diferentes presets
    presets = list(generator.preset_prompts.keys())
    
    # Limitar al número de presets disponibles
    count = min(count, len(presets))
    selected_presets = presets[:count]
    
    print(f"   Presets: {', '.join(selected_presets)}")
    
    saved_paths = []
    for i, preset in enumerate(selected_presets, 1):
        print(f"\n[{i}/{count}] Generando {preset}...")
        path = generator.quick_generate(preset)
        if path:
            saved_paths.append(path)
            print(f"   ✅ Guardado")
        else:
            print(f"   ❌ Falló")
    
    print(f"\n📊 Total generados: {len(saved_paths)}/{count}")
    return saved_paths

def main():
    """Función principal"""
    
    print("="*60)
    print("🤖 GENERADOR DE HUMANOS OPTIMIZADO")
    print("   v2.0 - Eficiencia mejorada")
    print("="*60)
    
    # Opciones
    print("\n🎯 MODOS:")
    print("   1. Interactivo (elige prompts)")
    print("   2. Rápido (genera ahora)")
    print("   3. Batch (varios a la vez)")
    print("   4. Probar todos los tipos")
    
    choice = input("\nSelecciona (1-4): ").strip()
    
    if choice == "1":
        generator = OptimizedHumanGenerator()
        generator.interactive_mode()
    
    elif choice == "2":
        saved_path = quick_start()
        if saved_path:
            print(f"\n📍 Ruta: {saved_path}")
    
    elif choice == "3":
        try:
            count = int(input("¿Cuántos generar? (2-5): ").strip())
            count = max(2, min(5, count))
            batch_generate(count)
        except:
            print("⚠️  Número inválido, generando 3")
            batch_generate(3)
    
    elif choice == "4":
        # Generar uno de cada tipo
        generator = OptimizedHumanGenerator()
        for preset in generator.preset_prompts.keys():
            print(f"\n🎭 Generando {preset}...")
            generator.quick_generate(preset)
    
    print("\n" + "="*60)
    print("✨ Proceso completado")

if __name__ == "__main__":
    # Configuración simple
    import signal
    signal.signal(signal.SIGINT, lambda s, f: print("\n\n⚠️  Interrumpido"))
    
    # Ejecutar
    main()