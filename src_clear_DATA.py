"""
Скрипт для поиска фотографий, которые не смогла угадать модель.
"""

import os
import cv2
import numpy as np
import torch
import timm
import shutil
import albumentations as A
from albumentations.pytorch import ToTensorV2
from glob import glob
from tqdm import tqdm
import torch.nn.functional as F

# --- КОНФИГУРАЦИЯ ---
MODEL_NAME = 'resnet50_1.pth' 
DATASET_PATH = 'lab1ImageClassification/train/train'  # Путь к данным
CHECKPOINT_PATH = 'lab1ImageClassification/checkpoints/' + MODEL_NAME # Путь где лежит модель
OUTPUT_DIR = 'suspicious_images'       # Куда сохранять неугаданные фото

IMG_SIZE = 300  # Размер фото на которых учились
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_to_idx = { 
    "Апельсин": 0, "Бананы": 1, "Груши": 2, "Кабачки": 3, "Капуста": 4,
    "Картофель": 5, "Киви": 6, "Лимон": 7, "Лук": 8, "Мандарины": 9,
    "Морковь": 10, "Огурцы": 11, "Томаты": 12, "Яблоки зеленые": 13, "Яблоки красные": 14 
}
idx_to_class = {v: k for k, v in class_to_idx.items()}

val_transforms = A.Compose([
    A.LongestMaxSize(max_size=IMG_SIZE),
    A.PadIfNeeded(
        min_height=IMG_SIZE,
        min_width=IMG_SIZE,
        border_mode=cv2.BORDER_CONSTANT,
        value=0,
        p=1.0
    ),
    A.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ToTensorV2(),
])

def find_noisy_labels():
    if os.path.exists(OUTPUT_DIR):
        shutil.rmtree(OUTPUT_DIR)
    os.makedirs(OUTPUT_DIR)

    # Загружаем модель
    print(f"Загрузка модели из {CHECKPOINT_PATH}...")
    model = timm.create_model('resnet50', pretrained=False, num_classes=15)
    
    # Загрузка весов
    state_dict = torch.load(CHECKPOINT_PATH, map_location=DEVICE)
    model.load_state_dict(state_dict)
    model.to(DEVICE)
    model.eval()

    print("Начинаем поиск ошибок в разметке...")
    
    # Сбор всех файлов
    all_images = []
    
    for class_name in os.listdir(DATASET_PATH):
        class_path = os.path.join(DATASET_PATH, class_name)
        if not os.path.isdir(class_path): continue
        
        if class_name not in class_to_idx: continue
        
        for subclass in os.listdir(class_path):
            sub_path = os.path.join(class_path, subclass)
            if not os.path.isdir(sub_path): continue
            
            imgs = glob(os.path.join(sub_path, '*.*')) 
            for img_path in imgs:
                all_images.append({
                    'path': img_path,
                    'true_label_name': class_name,
                    'true_label_idx': class_to_idx[class_name]
                })

    print(f"Всего изображений для проверки: {len(all_images)}")
    
    suspicious_count = 0
    
    # Проход по данным
    with torch.no_grad():
        for item in tqdm(all_images):
            image_path = item['path']
            true_label = item['true_label_idx']
            true_name = item['true_label_name']
            
            try:
                image = cv2.imdecode(np.fromfile(image_path, dtype=np.uint8), cv2.IMREAD_COLOR)
                if image is None: continue
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            except Exception as e:
                print(f"Error reading {image_path}: {e}")
                continue

            # Аугментация
            aug_img = val_transforms(image=image)['image']
            aug_img = aug_img.unsqueeze(0).to(DEVICE) # [1, 3, 300, 300]

            # Предсказание
            logits = model(aug_img)
            probs = F.softmax(logits, dim=1)
            
            # Получаем топ-1 класс и уверенность
            conf, pred_idx = torch.max(probs, dim=1)
            conf = conf.item()
            pred_idx = pred_idx.item()
            
            # Ищем ошибки
            if pred_idx != true_label:
                pred_name = idx_to_class[pred_idx]
                
                if conf > 0.5:
                    suspicious_count += 1
                    
                    save_folder = os.path.join(OUTPUT_DIR, true_name)
                    os.makedirs(save_folder, exist_ok=True)
                    
                    # Имя файла: [CONF_0.99]_PRED_Томаты_ORIG_image.jpg
                    filename = os.path.basename(image_path)
                    new_filename = f"[CONF_{conf:.2f}]_PRED_{pred_name}_{filename}"
                    
                    save_path = os.path.join(save_folder, new_filename)
                    shutil.copy(image_path, save_path)

    print(f"\nГотово! Найдено {suspicious_count} подозрительных изображений.")
    print(f"Проверь папку: {OUTPUT_DIR}")

find_noisy_labels()