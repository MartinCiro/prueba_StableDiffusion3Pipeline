import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
import cv2
import numpy as np
from PIL import Image
import os
import random
from datetime import datetime
from pathlib import Path

class HumanoidAvatarGenerator:
    def __init__(self, input_dir="./inputs"):
        print("🎭 Inicializando Generador de Avatares Humanoides...")
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"📱 Usando dispositivo: {self.device}")
        
        # Directorios
        self.input_dir = Path(input_dir)
        self.output_dir = Path("avatar_outputs")
        self.output_dir.mkdir(exist_ok=True)
        
        # Verificar inputs
        self.input_images = self.load_input_images()
        if not self.input_images:
            print("⚠️  No se encontraron imágenes en ./inputs/")
            print("   Por favor, añade selfies en formato JPG/PNG a la carpeta ./inputs/")
        
        # Modelos
        self.face_detector = None
        self.segmentation_model = None
        self.face_encoder = None
        
        # Transformaciones
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
        
        self.init_models()
    
    def load_input_images(self):
        """Carga todas las imágenes del directorio de inputs"""
        images = []
        valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff'}
        
        if self.input_dir.exists():
            for img_path in self.input_dir.glob('*'):
                if img_path.suffix.lower() in valid_extensions:
                    images.append(img_path)
        
        print(f"📸 Encontradas {len(images)} imágenes en {self.input_dir}")
        return images
    
    def init_models(self):
        """Inicializa los modelos necesarios"""
        print("📦 Cargando modelos...")
        
        # 1. Detector facial (usando OpenCV Haar Cascade o DNN)
        print("👁️  Cargando detector facial...")
        self.face_detector = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        )
        
        # 2. Modelo de segmentación (para separar persona/fondo)
        print("🎯 Cargando modelo de segmentación...")
        try:
            self.segmentation_model = models.segmentation.deeplabv3_resnet50(
                weights=models.segmentation.DeepLabV3_ResNet50_Weights.DEFAULT
            )
            self.segmentation_model.eval()
            self.segmentation_model.to(self.device)
        except Exception as e:
            print(f"   ⚠️  Error cargando modelo de segmentación: {e}")
            self.segmentation_model = None
        
        # 3. Codificador facial (para extraer características)
        print("🔍 Cargando codificador facial...")
        try:
            self.face_encoder = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT
            )
            num_features = self.face_encoder.fc.in_features
            self.face_encoder.fc = nn.Identity()
            self.face_encoder.eval()
            self.face_encoder.to(self.device)
        except Exception as e:
            print(f"   ⚠️  Error cargando codificador facial: {e}")
            self.face_encoder = None
        
        print("✅ Modelos cargados exitosamente")
    
    def detect_faces(self, image):
        """Detecta rostros en una imagen"""
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(
            gray, 
            scaleFactor=1.1, 
            minNeighbors=5, 
            minSize=(30, 30)
        )
        
        result = []
        for (x, y, w, h) in faces:
            result.append({
                'bbox': (x, y, w, h),
                'confidence': 1.0
            })
        
        return result
    
    def segment_person(self, image):
        """Segmenta la persona del fondo"""
        if self.segmentation_model is None:
            # Fallback: crear máscara simple basada en detección de piel
            h, w = image.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            
            # Convertir a HSV para detección de piel
            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
            
            # Rango para tonos de piel (ajustable)
            lower_skin = np.array([0, 20, 70], dtype=np.uint8)
            upper_skin = np.array([20, 255, 255], dtype=np.uint8)
            
            skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
            
            # Operaciones morfológicas para limpiar la máscara
            kernel = np.ones((5,5), np.uint8)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_CLOSE, kernel)
            skin_mask = cv2.morphologyEx(skin_mask, cv2.MORPH_OPEN, kernel)
            
            # Encontrar el contorno más grande (presumiblemente la persona)
            contours, _ = cv2.findContours(skin_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if contours:
                largest_contour = max(contours, key=cv2.contourArea)
                cv2.drawContours(mask, [largest_contour], -1, 255, -1)
            
            return mask
        
        # Usar modelo de segmentación DeepLabV3
        try:
            # Preprocesar imagen
            input_tensor = self.transform(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)))
            input_batch = input_tensor.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                output = self.segmentation_model(input_batch)['out'][0]
            
            # Obtener máscara de persona (clase 15 en COCO)
            person_mask = output.argmax(0).cpu().numpy()
            mask = (person_mask == 15).astype(np.uint8) * 255
            
            # Redimensionar a tamaño original
            mask = cv2.resize(mask, (image.shape[1], image.shape[0]))
            
            # Suavizar bordes
            mask = cv2.GaussianBlur(mask, (5, 5), 0)
            
            return mask
            
        except Exception as e:
            print(f"Error en segmentación: {e}")
            # Fallback
            h, w = image.shape[:2]
            return np.ones((h, w), dtype=np.uint8) * 255  # Máscara blanca completa
    
    def extract_face_features(self, face_image):
        """Extrae características (embedding) de un rostro"""
        if self.face_encoder is None or face_image.size == 0:
            return None
        
        try:
            # Asegurar que la imagen sea válida
            if face_image.shape[0] < 10 or face_image.shape[1] < 10:
                return None
                
            # Preprocesar
            face_resized = cv2.resize(face_image, (224, 224))
            face_pil = Image.fromarray(cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB))
            face_tensor = self.transform(face_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                features = self.face_encoder(face_tensor)
            
            return features.cpu().numpy().flatten()
            
        except Exception as e:
            print(f"Error extrayendo características: {e}")
            return None
    
    def create_humanoid_avatar(self, input_image):
        """Crea un avatar humanoide basado en una selfie"""
        print("   🔄 Procesando imagen...")
        
        try:
            # Leer imagen
            img = cv2.imread(str(input_image))
            if img is None:
                print(f"   ❌ Error leyendo imagen: {input_image}")
                return None
            
            # Redimensionar si es muy grande
            max_size = 1024
            h, w = img.shape[:2]
            if max(h, w) > max_size:
                scale = max_size / max(h, w)
                new_w = int(w * scale)
                new_h = int(h * scale)
                img = cv2.resize(img, (new_w, new_h))
            
            # 1. Detectar rostro
            faces = self.detect_faces(img)
            
            if not faces:
                print("   ⚠️  No se detectaron rostros, usando imagen completa")
                face_region = img
                face_features = None
            else:
                # Usar el rostro más grande
                best_face = max(faces, key=lambda x: x['bbox'][2] * x['bbox'][3])
                x, y, w, h = best_face['bbox']
                
                # Expandir región del rostro
                padding = int(min(w, h) * 0.3)
                x = max(0, x - padding)
                y = max(0, y - padding)
                w = min(img.shape[1] - x, w + 2*padding)
                h = min(img.shape[0] - y, h + 2*padding)
                
                face_region = img[y:y+h, x:x+w]
                print(f"   ✅ Rostro detectado: {w}x{h} píxeles")
                
                # Extraer características faciales
                face_features = self.extract_face_features(face_region)
            
            # 2. Segmentar persona
            print("   🎯 Segmentando...")
            person_mask = self.segment_person(img)
            
            # 3. Crear avatar
            print("   🎨 Generando avatar...")
            avatar = self.generate_humanoid_style(img, person_mask)
            
            # 4. Aplicar estilo final
            print("   ✨ Estilizando...")
            avatar_styled = self.apply_avatar_style(avatar)
            
            return avatar_styled
            
        except Exception as e:
            print(f"   ❌ Error procesando {input_image.name}: {str(e)}")
            return None
    
    def generate_humanoid_style(self, original_img, person_mask):
        """Genera una versión estilizada humanoide"""
        # Crear copia de la imagen original
        avatar = original_img.copy()
        
        # 1. Suavizar la imagen en general
        avatar = cv2.bilateralFilter(avatar, 9, 75, 75)
        
        # 2. Aplicar máscara de persona
        # Invertir máscara para obtener fondo
        background_mask = cv2.bitwise_not(person_mask)
        
        # Crear fondo blanco suave
        white_bg = np.full_like(avatar, 245)
        
        # Mezclar avatar con fondo blanco
        for c in range(3):
            avatar[:,:,c] = np.where(
                background_mask > 0,
                white_bg[:,:,c],
                avatar[:,:,c]
            )
        
        # 3. Suavizar bordes de la máscara
        kernel = np.ones((5,5), np.uint8)
        person_mask_smooth = cv2.GaussianBlur(person_mask, (7,7), 0)
        
        # Aplicar transición suave en bordes
        mask_float = person_mask_smooth.astype(float) / 255.0
        mask_3channel = np.stack([mask_float]*3, axis=2)
        
        # Mezcla suave
        white_bg_float = white_bg.astype(float)
        avatar_float = avatar.astype(float)
        
        avatar = (avatar_float * mask_3channel + white_bg_float * (1 - mask_3channel)).astype(np.uint8)
        
        # 4. Realzar características faciales si hay rostro detectado
        # Buscar área de piel para realzar
        hsv = cv2.cvtColor(avatar, cv2.COLOR_BGR2HSV)
        
        # Crear máscara de piel aproximada
        lower_skin = np.array([0, 30, 60])
        upper_skin = np.array([20, 150, 255])
        skin_mask = cv2.inRange(hsv, lower_skin, upper_skin)
        
        # Dilatar máscara de piel
        kernel = np.ones((3,3), np.uint8)
        skin_mask = cv2.dilate(skin_mask, kernel, iterations=2)
        
        # Suavizar piel
        avatar_blur = cv2.bilateralFilter(avatar, 15, 80, 80)
        
        # Aplicar piel suavizada solo en áreas de piel
        skin_mask_float = skin_mask.astype(float) / 255.0
        skin_mask_3channel = np.stack([skin_mask_float]*3, axis=2)
        
        avatar = (avatar_blur.astype(float) * skin_mask_3channel + 
                 avatar.astype(float) * (1 - skin_mask_3channel)).astype(np.uint8)
        
        # 5. Ajustar colores para estilo más vibrante
        # Aumentar saturación
        hsv = cv2.cvtColor(avatar, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.3, 0, 255).astype(np.uint8)
        avatar = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 6. Añadir contorno sutil
        edges = cv2.Canny(person_mask, 50, 150)
        edges = cv2.dilate(edges, np.ones((2,2), np.uint8), iterations=1)
        
        # Dibujar borde sutil
        avatar[edges > 0] = np.clip(avatar[edges > 0] * 0.8, 0, 255).astype(np.uint8)
        
        return avatar
    
    def apply_avatar_style(self, image):
        """Aplica filtros de estilo para hacerlo más 'avatar'"""
        styled = image.copy()
        
        # 1. Reducir ruido y suavizar
        styled = cv2.bilateralFilter(styled, 7, 50, 50)
        
        # 2. Aumentar contraste con CLAHE
        lab = cv2.cvtColor(styled, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        styled = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
        
        # 3. Ajustar brillo y saturación
        hsv = cv2.cvtColor(styled, cv2.COLOR_BGR2HSV)
        hsv[:,:,1] = np.clip(hsv[:,:,1] * 1.1, 0, 255)  # Saturación
        hsv[:,:,2] = np.clip(hsv[:,:,2] * 1.05, 0, 255)  # Brillo
        styled = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
        
        # 4. Filtro de nitidez suave
        kernel = np.array([[0, -1, 0],
                           [-1, 5, -1],
                           [0, -1, 0]])
        styled = cv2.filter2D(styled, -1, kernel)
        
        # 5. Reducir ligeramente la resolución para efecto estilizado
        if styled.shape[0] > 512:
            h, w = styled.shape[:2]
            scale = 512 / h
            new_w = int(w * scale)
            styled = cv2.resize(styled, (new_w, 512))
        
        return styled
    
    def process_all_inputs(self):
        """Procesa todas las imágenes en ./inputs y genera avatares"""
        if not self.input_images:
            print("❌ No hay imágenes para procesar")
            return []
        
        avatars = []
        successful = 0
        
        print(f"\n{'='*60}")
        print(f"PROCESANDO {len(self.input_images)} IMÁGENES")
        print(f"{'='*60}")
        
        for i, img_path in enumerate(self.input_images, 1):
            print(f"\n[{i}/{len(self.input_images)}] {img_path.name}")
            
            try:
                avatar = self.create_humanoid_avatar(img_path)
                
                if avatar is not None:
                    # Guardar avatar
                    output_path = self.output_dir / f"avatar_{img_path.stem}.jpg"
                    cv2.imwrite(str(output_path), avatar, [cv2.IMWRITE_JPEG_QUALITY, 95])
                    
                    avatars.append({
                        'original': str(img_path),
                        'avatar': str(output_path),
                        'image': avatar
                    })
                    
                    successful += 1
                    print(f"   ✅ Guardado: {output_path.name}")
                else:
                    print(f"   ❌ Falló: No se pudo procesar")
                    
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
        
        return avatars, successful
    
    def generate_composite_avatar(self, avatars):
        """Genera un avatar compuesto si hay múltiples exitosos"""
        if len(avatars) < 2:
            return None
        
        print("\n   🔀 Generando avatar compuesto...")
        
        try:
            # Redimensionar todas las imágenes al mismo tamaño
            target_size = (256, 256)
            resized_avatars = []
            
            for avatar_data in avatars:
                img = cv2.resize(avatar_data['image'], target_size)
                resized_avatars.append(img)
            
            # Crear promedio
            composite = np.zeros((target_size[1], target_size[0], 3), dtype=np.float32)
            
            for img in resized_avatars:
                composite += img.astype(np.float32) / len(resized_avatars)
            
            composite = composite.astype(np.uint8)
            
            # Aplicar estilo
            composite = self.apply_avatar_style(composite)
            
            # Guardar
            output_path = self.output_dir / "avatar_composite.jpg"
            cv2.imwrite(str(output_path), composite, [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            print(f"   ✅ Compuesto guardado: {output_path.name}")
            return composite
            
        except Exception as e:
            print(f"   ❌ Error generando compuesto: {e}")
            return None
    
    def display_results(self, avatars, successful, total):
        """Muestra los resultados"""
        print(f"\n{'='*60}")
        print("RESUMEN DE RESULTADOS")
        print(f"{'='*60}")
        
        print(f"\n📊 Estadísticas:")
        print(f"   • Total de imágenes: {total}")
        print(f"   • Procesadas exitosamente: {successful}")
        print(f"   • Tasa de éxito: {successful/total*100:.1f}%")
        
        if successful > 0:
            print(f"\n✅ Avatares generados en: {self.output_dir}/")
            
            if successful > 1:
                print("\n🔀 Avatar compuesto generado: avatar_composite.jpg")
            
            print(f"\n📋 Lista de avatares:")
            for i, avatar_data in enumerate(avatars[:5], 1):  # Mostrar solo primeros 5
                orig_name = Path(avatar_data['original']).name
                ava_name = Path(avatar_data['avatar']).name
                print(f"   {i}. {orig_name} → {ava_name}")
            
            if len(avatars) > 5:
                print(f"   ... y {len(avatars) - 5} más")
            
            print(f"\n💡 Para descargar:")
            print(f"   scp -r {self.output_dir}/*.jpg usuario@tu-local:~/")
            
        else:
            print("\n❌ No se generaron avatares exitosamente")
            print("\n🔧 Solución de problemas:")
            print("   1. Verifica que las imágenes en ./inputs/ sean válidas")
            print("   2. Asegúrate de que sean fotos de rostros claramente visibles")
            print("   3. Prueba con imágenes bien iluminadas y frontal")
            print("   4. Verifica que OpenCV esté instalado correctamente")

def main():
    """Función principal"""
    print("="*60)
    print("GENERADOR DE AVATARES HUMANOIDES")
    print("Basado en Selfies Reales")
    print("="*60)
    
    # Inicializar generador
    generator = HumanoidAvatarGenerator(input_dir="./inputs")
    
    # Procesar todas las imágenes
    avatars, successful = generator.process_all_inputs()
    
    # Generar avatar compuesto si hay múltiples exitosos
    if successful > 1:
        generator.generate_composite_avatar(avatars)
    
    # Mostrar resumen
    generator.display_results(avatars, successful, len(generator.input_images))
    
    # Si no hay imágenes, mostrar ayuda
    if not generator.input_images:
        print("\n" + "="*60)
        print("📁 INSTRUCCIONES DE USO")
        print("="*60)
        print("\n1. Crea un directorio llamado 'inputs' en la misma carpeta:")
        print("   mkdir inputs")
        print("\n2. Copia tus selfies (fotos de rostro) al directorio:")
        print("   cp /ruta/tus/fotos/*.jpg ./inputs/")
        print("\n3. Ejecuta el generador:")
        print("   python main.py")
        print("\n4. Encuentra tus avatares en ./avatar_outputs/")
        print("\n💡 Recomendaciones:")
        print("   • Usa fotos con buen enfoque y luz")
        print("   • Rostros frontales funcionan mejor")
        print("   • Imágenes JPG/PNG de al menos 500x500 px")

if __name__ == "__main__":
    main()