import io
import os

import mysql.connector
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from werkzeug.security import generate_password_hash, check_password_hash

app = Flask(__name__)

# Reject uploads larger than 10 MB
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024

# Comma separated list of allowed origins, e.g. "http://localhost:3000"
CORS(app, origins=os.environ.get('CORS_ORIGINS', '*').split(','))

model = load_model('model.h5')

# Class order produced by pd.Categorical(skin_df['cell_type']).codes during training
# (alphabetical order of the lesion type names)
CLASS_NAMES = [
    'Actinic keratoses',
    'Basal cell carcinoma',
    'Benign keratosis-like lesions',
    'Dermatofibroma',
    'Melanocytic nevi',
    'Melanoma',
    'Vascular lesions',
]
MELANOMA_INDEX = CLASS_NAMES.index('Melanoma')

# Training normalised images with (x - x_train_mean) / x_train_std.
# Set these to the values printed during training. If they are not set,
# each image is standardised with its own mean and std as an approximation.
TRAIN_MEAN = os.environ.get('MODEL_TRAIN_MEAN')
TRAIN_STD = os.environ.get('MODEL_TRAIN_STD')

# Training resized images with .resize((100, 75)) -> width 100, height 75
IMAGE_SIZE = (100, 75)


def get_db():
    return mysql.connector.connect(
        host=os.environ.get('DB_HOST', 'localhost'),
        user=os.environ.get('DB_USER', 'root'),
        password=os.environ.get('DB_PASSWORD', ''),
        database=os.environ.get('DB_NAME', 'cancer_detection_project_db'),
    )


def get_credentials():
    data = request.get_json(silent=True) or {}
    username = data.get('username')
    password = data.get('password')
    if not username or not password:
        return None, None
    return username, password


def password_matches(stored, password):
    # Hashed passwords created by /signup
    if stored.startswith(('pbkdf2:', 'scrypt:')):
        return check_password_hash(stored, password)
    # Legacy rows stored in plain text before hashing was added
    return stored == password


@app.route('/login', methods=['POST'])
def login():
    username, password = get_credentials()
    if username is None:
        return jsonify({"message": "Username and password are required"}), 400

    db = get_db()
    try:
        cursor = db.cursor()
        cursor.execute("SELECT password FROM users WHERE username = %s", (username,))
        user = cursor.fetchone()
        cursor.close()
    finally:
        db.close()

    if user and password_matches(user[0], password):
        return jsonify({"message": "Login successful", "username": username})
    return jsonify({"message": "Login failed"}), 401


# Define a route for user signup
@app.route('/signup', methods=['POST'])
def signup():
    username, password = get_credentials()
    if username is None:
        return jsonify({"message": "Username and password are required"}), 400

    db = get_db()
    try:
        cursor = db.cursor()

        # Check if the username is already taken
        cursor.execute("SELECT 1 FROM users WHERE username = %s", (username,))
        if cursor.fetchone():
            cursor.close()
            return jsonify({"message": "Username already exists"}), 400

        # Insert the new user into the database
        cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)",
                       (username, generate_password_hash(password)))
        db.commit()
        cursor.close()
    finally:
        db.close()

    return jsonify({"message": "Signup successful"})


ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'jfif', 'png'}


# Function to check if a file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def preprocess(img_bytes):
    img = Image.open(io.BytesIO(img_bytes)).convert('RGB')
    img = img.resize(IMAGE_SIZE)
    img = np.asarray(img, dtype=np.float32)  # shape (75, 100, 3)
    if TRAIN_MEAN is not None and TRAIN_STD is not None:
        img = (img - float(TRAIN_MEAN)) / float(TRAIN_STD)
    else:
        img = (img - img.mean()) / (img.std() + 1e-7)
    return img[np.newaxis, ...]


def predict(img_bytes):
    probabilities = model.predict(preprocess(img_bytes))[0]
    best = int(np.argmax(probabilities))
    return {
        "message": 'Melanoma' if best == MELANOMA_INDEX else 'Non-Melanoma',
        "prediction": CLASS_NAMES[best],
        "confidence": float(probabilities[best]),
        "melanoma_probability": float(probabilities[MELANOMA_INDEX]),
    }


# Define the image upload route
@app.route('/upload', methods=['POST'])
def upload_image():
    # Check if the POST request has a file part
    if 'file' not in request.files:
        return jsonify({"message": "No file part"}), 400

    file = request.files['file']

    # If the user does not select a file, the browser submits an empty part without a filename
    if file.filename == '':
        return jsonify({"message": "No selected file"}), 400

    # Check if the file extension is allowed
    if not allowed_file(file.filename):
        return jsonify({"message": "Invalid file type"}), 400

    try:
        result = predict(file.read())
    except UnidentifiedImageError:
        return jsonify({"message": "Invalid image"}), 400

    return jsonify(result)


@app.route("/imageUpload", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files.get("image")
        if image_file and image_file.filename:
            try:
                result = predict(image_file.read())
            except UnidentifiedImageError:
                return render_template("index.html", prediction="Invalid image"), 400
            return render_template("index.html", prediction=result["prediction"])
    return render_template("index.html", prediction=None)


if __name__ == '__main__':
    app.run(host=os.environ.get('HOST', '127.0.0.1'),
            port=int(os.environ.get('PORT', 12000)),
            debug=os.environ.get('FLASK_DEBUG') == '1')
