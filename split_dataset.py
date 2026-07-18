"""
Script untuk membagi dataset mentah (folder per kelas) menjadi
struktur dataset/train dan dataset/test secara otomatis (80:20).

Jalankan SEKALI SAJA sebelum training.

Sebelum menjalankan, sesuaikan SOURCE_DIR di bawah ini
dengan lokasi folder dataset mentah kamu, contoh:
    archive/sneakers-dataset
"""

import os
import shutil
import random

# ==========================================
# GANTI SESUAI LOKASI DATASET MENTAH KAMU
# ==========================================
SOURCE_DIR = "dataset/sneakers-dataset/sneakers-dataset"

TRAIN_DIR = "dataset/train"
TEST_DIR = "dataset/test"
SPLIT_RATIO = 0.8  # 80% train, 20% test

random.seed(42)

if not os.path.exists(SOURCE_DIR):
    raise FileNotFoundError(
        f"Folder '{SOURCE_DIR}' tidak ditemukan. "
        f"Cek nama folder dataset mentah kamu dan ubah SOURCE_DIR di script ini."
    )

os.makedirs(TRAIN_DIR, exist_ok=True)
os.makedirs(TEST_DIR, exist_ok=True)

class_folders = [
    f for f in os.listdir(SOURCE_DIR)
    if os.path.isdir(os.path.join(SOURCE_DIR, f))
]

print(f"Ditemukan {len(class_folders)} kelas: {class_folders}")

for class_name in class_folders:
    src_class_dir = os.path.join(SOURCE_DIR, class_name)
    images = [
        f for f in os.listdir(src_class_dir)
        if f.lower().endswith(('.jpg', '.jpeg', '.png'))
    ]
    random.shuffle(images)

    split_index = int(len(images) * SPLIT_RATIO)
    train_images = images[:split_index]
    test_images = images[split_index:]

    train_class_dir = os.path.join(TRAIN_DIR, class_name)
    test_class_dir = os.path.join(TEST_DIR, class_name)
    os.makedirs(train_class_dir, exist_ok=True)
    os.makedirs(test_class_dir, exist_ok=True)

    for img in train_images:
        shutil.copy(
            os.path.join(src_class_dir, img),
            os.path.join(train_class_dir, img)
        )
    for img in test_images:
        shutil.copy(
            os.path.join(src_class_dir, img),
            os.path.join(test_class_dir, img)
        )

    print(f"{class_name}: {len(train_images)} train, {len(test_images)} test")

print("\nSelesai! Dataset sudah terbagi di folder 'dataset/train' dan 'dataset/test'.")