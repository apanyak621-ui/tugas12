import os
import json
import numpy as np
import urllib.request
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# Download Model TFLite dari Hugging Face jika belum ada secara lokal
# (versi TFLite jauh lebih kecil & hemat RAM dibanding .h5 penuh)
# ==========================================
MODEL_URL = 'https://huggingface.co/ininyo/Tugas-12/resolve/main/xception_sneakers.tflite'
MODEL_PATH = 'model/xception_sneakers.tflite'
LABELS_PATH = 'model/class_labels.json'

os.makedirs('model', exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Model belum ada, mendownload dari Hugging Face...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download model selesai.")

interpreter = tf.lite.Interpreter(model_path=MODEL_PATH)
interpreter.allocate_tensors()
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

with open(LABELS_PATH, 'r') as f:
    class_labels = json.load(f)  # contoh: {"0": "adidas", "1": "converse", ...}


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0).astype(np.float32) / 255.0

    interpreter.set_tensor(input_details[0]['index'], img_array)
    interpreter.invoke()
    prediction = interpreter.get_tensor(output_details[0]['index'])[0]

    predicted_index = int(np.argmax(prediction))
    confidence = float(np.max(prediction)) * 100
    predicted_label = class_labels[str(predicted_index)]

    return predicted_label, confidence


@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', error='Tidak ada file yang diunggah.')

    file = request.files['file']
    if file.filename == '':
        return render_template('index.html', error='Silakan pilih gambar terlebih dahulu.')

    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    label, confidence = predict_image(filepath)

    return render_template(
        'result.html',
        label=label,
        confidence=round(confidence, 2),
        image_path=filepath
    )


if __name__ == '__main__':
    app.run(debug=True)