"""
Training Model Klasifikasi Jenis Sneakers menggunakan Transfer Learning Xception
Struktur dataset yang dibutuhkan:

dataset/
├── train/
│   ├── nike/
│   ├── adidas/
│   ├── converse/
│   ├── vans/
│   └── new_balance/
└── test/
    ├── nike/
    ├── adidas/
    ├── converse/
    ├── vans/
    └── new_balance/
"""

import tensorflow as tf
from tensorflow.keras.applications import Xception
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import json
import os

# Cek versi TensorFlow
print(tf.__version__)

# Buat folder model/ jika belum ada
os.makedirs('model', exist_ok=True)

# ==========================================
# 1. Load Model Xception Pretrained
# ==========================================
base_model = Xception(weights='imagenet', include_top=False, input_shape=(299, 299, 3))

# Membekukan semua layer agar tidak di-train ulang
for layer in base_model.layers:
    layer.trainable = False

# ==========================================
# 2. Tambahkan Layer Klasifikasi Baru
# ==========================================
# ==========================================
# 3. Persiapkan Dataset (dijalankan dulu agar jumlah kelas terdeteksi otomatis)
# ==========================================
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    zoom_range=0.2,
    horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_data = train_datagen.flow_from_directory(
    'dataset/train',
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

test_data = test_datagen.flow_from_directory(
    'dataset/test',
    target_size=(299, 299),
    batch_size=32,
    class_mode='categorical'
)

# Jumlah kelas dideteksi otomatis dari jumlah folder di dataset/train
JUMLAH_KELAS = train_data.num_classes
print(f"Jumlah kelas terdeteksi: {JUMLAH_KELAS}")

model = Sequential([
    base_model,  # Model Xception
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.5),
    Dense(JUMLAH_KELAS, activation='softmax')  # Output multi-class
])

# Compile model dengan learning rate lebih kecil agar model belajar stabil
model.compile(optimizer=Adam(learning_rate=0.0001), loss='categorical_crossentropy', metrics=['accuracy'])

# Tampilkan arsitektur model
model.summary()

# Simpan mapping label kelas (dibutuhkan saat prediksi di Flask)
class_indices = train_data.class_indices
labels = {v: k for k, v in class_indices.items()}
with open('model/class_labels.json', 'w') as f:
    json.dump(labels, f)
print("Label kelas:", labels)

# ==========================================
# 4. Latih Model
# ==========================================
early_stop = EarlyStopping(
    monitor='val_accuracy',
    patience=5,
    restore_best_weights=True
)

history = model.fit(
    train_data,
    validation_data=test_data,
    epochs=30,
    callbacks=[early_stop]
)

# ==========================================
# 5. Evaluasi Model
# ==========================================
test_loss, test_acc = model.evaluate(test_data)
print(f'Akurasi Model: {test_acc:.2f}')

# ==========================================
# 6. Simpan Model
# ==========================================
model.save('model/xception_sneakers.h5')
print("Model berhasil disimpan di model/xception_sneakers.h5")
