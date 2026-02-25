import cv2
import os
import numpy as np
from glob import glob
from tqdm import tqdm

def process_photos_natural(input_folder, output_folder, mean_size, std_size, min_side_limit=300):
    """
    mean_size, std_size: статистика твоего датасета (например, 350 и 50)
    min_side_limit: жесткое ограничение, чтобы сторона не стала меньше 300px (под модель)
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    image_paths = []
    for ext in ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']:
        image_paths.extend(glob(os.path.join(input_folder, ext)))

    print(f"Найдено {len(image_paths)} фото. Обработка...")

    for path in tqdm(image_paths):
        # 1. Читаем исходное фото
        try:
            img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        except Exception:
            img = None
            
        if img is None: continue

        orig_h, orig_w = img.shape[:2]
        
        # --- ЭТАП 1: СЛУЧАЙНЫЙ КРОП (Вырезаем кусок) ---
        
        # Сколько площади оригинала мы хотим оставить? (от 60% до 100%)
        # Это симулирует разное расстояние до объекта (зум)
        area_scale = np.random.uniform(0.6, 1.0)
        target_area = (orig_h * orig_w) * area_scale
        
        # Какое соотношение сторон мы хотим? (от 3:4 до 4:3)
        # Это делает фото не квадратными. 
        # log_uniform распределение, чтобы 0.75 и 1.33 были равновероятны
        aspect_ratio = np.exp(np.random.uniform(np.log(0.75), np.log(1.33)))
        
        # Вычисляем размеры кропа
        crop_w = int(np.sqrt(target_area * aspect_ratio))
        crop_h = int(np.sqrt(target_area / aspect_ratio))
        
        # Проверяем, чтобы кроп не вылезал за границы оригинала
        if crop_w > orig_w: crop_w = orig_w
        if crop_h > orig_h: crop_h = orig_h
        
        # Выбираем случайную позицию кропа
        x = np.random.randint(0, orig_w - crop_w + 1)
        y = np.random.randint(0, orig_h - crop_h + 1)
        
        # Вырезаем!
        crop_img = img[y:y+crop_h, x:x+crop_w]
        
        # --- ЭТАП 2: СЖАТИЕ ДО НУЖНОГО РАЗМЕРА (Ресайз) ---
        
        # Генерируем "целевой размер" (эффективную сторону) на основе статистики датасета
        # Например, датасет в среднем 350px. Генерируем число ~ N(350, 50)
        target_effective_side = np.random.normal(mean_size, std_size)
        
        # Жесткое ограничение снизу: нам не нужны фото меньше 300px для EfficientNetV2
        target_effective_side = max(target_effective_side, min_side_limit)
        
        # Текущая эффективная сторона кропа
        current_effective_side = np.sqrt(crop_w * crop_h)
        
        # Вычисляем коэффициент уменьшения
        scale_factor = target_effective_side / current_effective_side
        
        # Новые размеры
        new_w = int(crop_w * scale_factor)
        new_h = int(crop_h * scale_factor)
        
        # Финальный ресайз
        resized_img = cv2.resize(crop_img, (new_w, new_h), interpolation=cv2.INTER_AREA)

        # Сохранение
        filename = os.path.basename(path)
        save_path = os.path.join(output_folder, filename)
        
        success, encoded_img = cv2.imencode('.jpg', resized_img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
        if success:
            encoded_img.tofile(save_path)

    print(f"Готово! Сохранено в {output_folder}")

process_photos_natural(
    input_folder='kiwi', 
    output_folder='3001', 
    mean_size=175.41173968888586, 
    std_size=61.02524048352011,
    min_side_limit=50
)