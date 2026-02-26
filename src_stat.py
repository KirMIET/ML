"""
Скрипт для подсчета статистики датасета
"""

import numpy as np
import os
from glob import glob
from PIL import Image
from tqdm import tqdm

def get_dataset_stats(dataset_path):
    print("Подсчет статистики текущего датасета...")
    effective_sides = []

    files = glob(os.path.join(dataset_path, "**", "*.*"), recursive=True)
    images = [f for f in files if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    for img_path in tqdm(images):
        try:
            with Image.open(img_path) as img:
                w, h = img.size
                side = np.sqrt(w * h)
                effective_sides.append(side)
        except:
            pass
    
    effective_sides = np.array(effective_sides)
    mean_side = np.mean(effective_sides)
    std_side = np.std(effective_sides)

    print(f"\nСтатистика датасета:")
    print(f"Средний размер стороны (Mean): {mean_side:.2f} px")
    print(f"Стандартное отклонение (Std): {std_side:.2f} px")
    
    return mean_side, std_side

# Укажи путь к папке train
dataset_mean, dataset_std = get_dataset_stats('lab1ImageClassification/train/train')

print(dataset_mean, dataset_std)