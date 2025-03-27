# Install required libraries
!pip install flask pyngrok tensorflow pillow

import os
from flask import Flask, request, render_template, flash, redirect, url_for
from werkzeug.utils import secure_filename
import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
import numpy as np
from pyngrok import ngrok
from google.colab import files

# Initialize Flask app
app = Flask(__name__, template_folder='/content/templates')
app.secret_key = "supersecretkey"
UPLOAD_FOLDER = '/content/static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure directories exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs('/content/templates', exist_ok=True)
os.makedirs('/content/static', exist_ok=True)

# Load or create InceptionV3 model
MODEL_PATH = "/content/inception_model.h5"

def create_inceptionv3_model():
    # Load base InceptionV3 model
    base_model = InceptionV3(weights='imagenet', include_top=False, input_shape=(299, 299))
    
    # Add custom layers
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    # Create final model
    model = Model(inputs=base_model.input, outputs=predictions)
    
    # Freeze base model layers
    for layer in base_model.layers:
        layer.trainable = False
    
    return model

# Load or initialize model
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
else:
    print("Model not found. Creating new InceptionV3 model...")
    model = create_inceptionv3_model()
    print("Please upload your trained model weights if available.")
    uploaded = files.upload()
    if uploaded:
        for fn in uploaded.keys():
            os.rename(fn, MODEL_PATH)
        model = tf.keras.models.load_model(MODEL_PATH)

# File extension checker
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Image preprocessing for InceptionV3
def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(299, 299))  # InceptionV3 requires 299x299
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)  # InceptionV3 specific preprocessing
    return img_array

# Routes
@app.route('/', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(request.url)
    
    file = request.files['file']
    
    if file.filename == '':
        flash('No selected file')
        return redirect(request.url)
    
    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(file_path)

        # Preprocess and predict
        img_array = preprocess_image(file_path)
        prediction = model.predict(img_array)[0][0]
        label = "Fresh" if prediction > 0.5 else "Defect"
        confidence = prediction if prediction > 0.5 else 1 - prediction

        return render_template('result.html', 
                             filename=filename, 
                             prediction=label, 
                             confidence=f"{confidence:.4f}")
    else:
        flash('Allowed file types are png, jpg, jpeg')
        return redirect(request.url)

# HTML templates with beautiful interface
with open('/content/templates/index.html', 'w') as f:
    f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Dragon Fruit Classifier</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding-top: 50px;
        }
        .card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .btn-primary {
            background-color: #ff1493;
            border-color: #ff1493;
            transition: all 0.3s;
        }
        .btn-primary:hover {
            background-color: #c71585;
            border-color: #c71585;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-6">
                <div class="card p-4 mt-5">
                    <h1 class="text-center mb-4" style="color: #ff1493;">Dragon Fruit Classifier</h1>
                    <p class="text-center text-muted mb-4">Upload an image to determine if your dragon fruit is fresh or defective</p>
                    <form method="POST" action="/predict" enctype="multipart/form-data">
                        <div class="mb-3">
                            <input type="file" class="form-control" name="file" accept="image/*" required>
                        </div>
                        <div class="d-grid">
                            <input type="submit" value="Analyze Fruit" class="btn btn-primary btn-lg">
                        </div>
                    </form>
                    {% with messages = get_flashed_messages() %}
                        {% if messages %}
                            {% for message in messages %}
                                <div class="alert alert-danger mt-3" role="alert">
                                    {{ message }}
                                </div>
                            {% endfor %}
                        {% endif %}
                    {% endwith %}
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>''')

with open('/content/templates/result.html', 'w') as f:
    f.write('''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Prediction Result</title>
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/css/bootstrap.min.css" rel="stylesheet">
    <style>
        body {
            background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
            min-height: 100vh;
            padding-top: 50px;
        }
        .card {
            border: none;
            border-radius: 15px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        }
        .btn-primary {
            background-color: #ff1493;
            border-color: #ff1493;
        }
        .btn-primary:hover {
            background-color: #c71585;
            border-color: #c71585;
        }
        .prediction-text {
            color: #ff1493;
            font-weight: bold;
        }
        .img-preview {
            border-radius: 10px;
            border: 2px solid #ff1493;
        }
    </style>
</head>
<body>
    <div class="container">
        <div class="row justify-content-center">
            <div class="col-md-8">
                <div class="card p-4 mt-5">
                    <h1 class="text-center mb-4" style="color: #ff1493;">Prediction Result</h1>
                    <div class="text-center mb-4">
                        <img src="/static/uploads/{{ filename }}" alt="Uploaded Image" class="img-preview" style="max-width: 100%; height: auto; max-height: 400px;">
                    </div>
                    <div class="text-center">
                        <h3>Prediction: <span class="prediction-text">{{ prediction }}</span></h3>
                        <p class="text-muted">Confidence: {{ confidence }}</p>
                    </div>
                    <div class="d-grid mt-4">
                        <a href="/" class="btn btn-primary btn-lg">Analyze Another Fruit</a>
                    </div>
                </div>
            </div>
        </div>
    </div>
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.3.0/dist/js/bootstrap.bundle.min.js"></script>
</body>
</html>''')

# Set up ngrok
!ngrok authtoken '2rRUQEGHobLlKiAq03dSqiT2MLM_3UJ9UJTWmaufCv9YtvyTi'
public_url = ngrok.connect(5000)
print(f"Public URL: {public_url}")

# Run the app
app.run(port=5000)
