import cv2
import os
from glob import glob
from tqdm import tqdm

def resize_my_photos(input_folder, output_folder, size=(400, 400)):
    # Создаем папку, если её нет
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # Ищем все картинки (jpg, png, jpeg)
    extensions = ['*.jpg', '*.jpeg', '*.png']
    image_paths = []
    for ext in extensions:
        image_paths.extend(glob(os.path.join(input_folder, ext)))

    print(f"Найдено {len(image_paths)} фото. Начинаю обработку...")

    for path in tqdm(image_paths):
        import numpy as np
        img = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_COLOR)
        
        if img is None:
            print(f"Ошибка чтения: {path}")
            continue

        resized_img = cv2.resize(img, size, interpolation=cv2.INTER_AREA)

        # Сохраняем результат
        file_name = os.path.basename(path)
        save_path = os.path.join(output_folder, file_name)
        
        # Записываем файл
        success, encoded_img = cv2.imencode('.jpg', resized_img)
        if success:
            encoded_img.tofile(save_path)

    print(f"Готово! Все фото сохранены в {output_folder}")

# --- ИСПОЛЬЗОВАНИЕ ---
# Укажи путь к своим новым фото из магазина
resize_my_photos(
    input_folder='D:\\LabsMIET\\MLlab\\lab1ImageClassification\\train\\train\\Груши\\2001', 
    output_folder='D:\\LabsMIET\\MLlab\\lab1ImageClassification\\train\\train\\Груши\\2002'
)