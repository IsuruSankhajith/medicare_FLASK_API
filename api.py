import os
import torch
from flask import Flask, request, jsonify
from flask import request
from flask import render_template

import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import io
from PIL import Image
import mysql.connector
from flask_cors import CORS

app = Flask(__name__)

CORS(app)

model = load_model('model.h5')
model.make_predict_function()

data = pd.read_csv('HAM10000_metadata.csv')

# MySQL Configuration
db = mysql.connector.connect(
    host="localhost",
    user="root",
    password="1998",
    database="cancer_detection_project_db"
)

@app.route('/login', methods=['POST'])
def login():
    data = request.get_json()
    username = data['username']
    password = data['password']

    cursor = db.cursor()

    cursor.execute("SELECT * FROM users WHERE username = %s AND password = %s", (username, password))
    user = cursor.fetchone()

    if user:
        return jsonify({"message": "Login successful", "username": username})
    else:
        return jsonify({"message": "Login failed"})

# Define a route for user signup
@app.route('/signup', methods=['POST'])
def signup():
    data = request.get_json()
    username = data['username']
    password = data['password']

    cursor = db.cursor()

    # Check if the username is already taken
    cursor.execute("SELECT * FROM users WHERE username = %s", (username,))
    existing_user = cursor.fetchone()

    if existing_user:
        return jsonify({"message": "Username already exists"}), 400

    # Insert the new user into the database
    cursor.execute("INSERT INTO users (username, password) VALUES (%s, %s)", (username, password))
    db.commit()

    return jsonify({"message": "Signup successful"})

# Specify the directory to store uploaded images
ALLOWED_EXTENSIONS = {'jpg', 'jpeg', 'jfif'}

# Function to check if a file extension is allowed
def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

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

    # Read image data from the FileStorage object
    img_bytes = file.read()

    # Convert image data to a PIL image
    img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))

    # Preprocess the image
    # img = image.load_img(image_file, target_size=(224, 224))
    img = Image.open(io.BytesIO(img_bytes))
    img = img.resize((75, 100))
    img = image.img_to_array(img)
    img = img / 255.0  # Normalize the image
    img = img.reshape((1, 75, 100, 3))

    # Make predictions using your model
    predictions = model.predict(img)
    class_name = 'Melanoma' if predictions[0][0] < 0.015 else 'Non-Melanoma'

    return jsonify({"message": class_name})

@app.route("/imageUpload", methods=["GET", "POST"])
def upload_predict():
    if request.method == "POST":
        image_file = request.files["image"]
        if image_file:
            # image_location = os.path.join(
            #     UPLOAD_FOLDER,
            #     image_file.filename
            # )
            # image_file.save(image_location)

            # Read image data from the FileStorage object
            img_bytes = image_file.read()

            # Convert image data to a PIL image
            img = image.load_img(io.BytesIO(img_bytes), target_size=(224, 224))

            # Preprocess the image
            # img = image.load_img(image_file, target_size=(224, 224))
            img = Image.open(io.BytesIO(img_bytes))
            img = img.resize((75, 100))
            img = image.img_to_array(img)
            img = img / 255.0  # Normalize the image
            img = img.reshape((1, 75, 100, 3))

            # Make predictions using your model
            predictions = model.predict(img)
            class_name = 'Melanoma' if predictions[0][0] < 0.015 else 'Non-Melanoma'

            return render_template("index.html", prediction=class_name)
    return render_template("index.html", prediction=0)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=12000, debug=True)
