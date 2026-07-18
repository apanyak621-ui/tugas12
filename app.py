import os
import json
import numpy as np
import urllib.request
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from werkzeug.utils import secure_filename

app = Flask(__name__)

UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==========================================
# Download Model dari Hugging Face jika belum ada secara lokal
# (model terlalu besar untuk disimpan langsung di GitHub)
# ==========================================
MODEL_URL = 'https://huggingface.co/ininyo/Tugas-12/resolve/main/xception_sneakers.h5'
MODEL_PATH = 'model/xception_sneakers.h5'
LABELS_PATH = 'model/class_labels.json'

os.makedirs('model', exist_ok=True)

if not os.path.exists(MODEL_PATH):
    print("Model belum ada, mendownload dari Hugging Face...")
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Download model selesai.")

model = load_model(MODEL_PATH)

with open(LABELS_PATH, 'r') as f:
    class_labels = json.load(f)  # contoh: {"0": "adidas", "1": "converse", ...}


def predict_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0

    prediction = model.predict(img_array)
    predicted_index = int(np.argmax(prediction[0]))
    confidence = float(np.max(prediction[0])) * 100
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