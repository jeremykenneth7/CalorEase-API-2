import io
import os
import re
from flask import Flask, jsonify, request
from werkzeug.utils import secure_filename
import tensorflow as tf
from PIL import Image
import numpy as np
from collections import Counter
from google.cloud import storage

app = Flask(__name__)
app.config["ALLOWED_EXTENSIONS"] = {'png', 'jpg', 'jpeg'}

bucket_name = 'imagestoragedatabase'
os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = './calorease-c3cd0-2778d4b3c4bd.json'
client = storage.Client()

model_cat = ["daging", "jajanan", "karbo", "lauk", "olahan_daging", "sayur"]
models = {}
category_indices = {}
threshold = 0.65

def read_label_file(label_path):
    id_pattern = r'id:\s*(\d+)'
    display_name_pattern = r'display_name:\s*"([^"]*)"'
    with open(label_path, 'r') as file:
        pbtxt_content = file.read()
        ids = [int(i) for i in re.findall(id_pattern, pbtxt_content)]
        display_names = re.findall(display_name_pattern, pbtxt_content)
    result = {}
    for i in range(len(display_names)):
        result[ids[i]] = {'id': ids[i], 'name': display_names[i]}
    return result

for cat in model_cat:
    saved_model_path = os.path.join("custom_model_lite", f"{cat}_efficientdet_d0", "saved_model")
    models[cat] = tf.saved_model.load(saved_model_path)
    label_map_path = os.path.join("custom_model_lite", f"{cat}_efficientdet_d0", "food_label_map.pbtxt")
    category_indices[cat] = read_label_file(label_map_path)

def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in app.config["ALLOWED_EXTENSIONS"]

@app.route("/")
def index():
    return jsonify({
        "status": {
            "code": 200,
            "message": "Success fetching the API",
        },
        "data": None
    }), 200

@app.route("/upload", methods=['POST'])
def upload():
    if request.method == "POST":
        if 'image' not in request.files:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "No image sent"
                },
                "data": None
            }), 400

        image = request.files['image']
        if image and allowed_file(image.filename):
            filename = secure_filename(image.filename)
            bucket = client.bucket(bucket_name)
            blob = bucket.blob(filename)
            blob.upload_from_file(image)

            # Create GCS path
            image_gcs_path = f"gs://{bucket_name}/{filename}"

            return jsonify({
                "status": {
                    "code": 200,
                    "message": "Image uploaded to GCS",
                },
                "data": {
                    "image_path": image_gcs_path
                }
            }), 200
        else:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Bad request or unsupported file type"
                },
                "data": None,
            }), 400

@app.route("/prediction", methods=['POST'])
def prediction():
    if request.method == "POST":
        if 'image_path' not in request.json:
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Bad request or missing image path"
                },
                "data": None,
            }), 400

        image_path = request.json['image_path']

        # Fetch the image from GCS using the provided image_path
        bucket_name = 'imagestoragedatabase'
        blob_name = image_path.split('gs://{}/'.format(bucket_name))[1]
        blob = client.bucket(bucket_name).get_blob(blob_name)

        if not blob:  # Check if blob exists
            return jsonify({
                "status": {
                    "code": 400,
                    "message": "Image not found in storage"
                },
                "data": None,
            }), 400
            
        # Download the image from GCS to memory (BytesIO)
        image_content = io.BytesIO()
        blob.download_to_file(image_content)
        image_content.seek(0)  # Reset the file pointer to the beginning

        # Perform inference on the image content
        img_np = np.array(Image.open(image_content))
        input_tensor = tf.convert_to_tensor(img_np, dtype=tf.uint8)
        input_tensor = input_tensor[tf.newaxis, ...]
        input_tensor = input_tensor[:, :, :, :3]

        obj_detected = []
        for cat in model_cat:
            detections = models[cat](input_tensor)
            classes = detections['detection_classes'][0].numpy()
            scores = detections['detection_scores'][0].numpy()
            for i in range(len(scores)):
                if (scores[i] > threshold) and (scores[i] <= 1.0):
                    object_name = category_indices[cat][classes[i]]['name']
                    obj_detected.append(object_name)
        detections_dict = dict(Counter(obj_detected))
        
        # Format predictions
        formatted_predictions = []
        idx = 1
        for name, count in detections_dict.items():
                formatted_prediction = {
                    "id": str(idx),
                    "nama": name,
                    "jumlah": str(count)
                }
                formatted_predictions.append(formatted_prediction)
                idx += 1

        return jsonify({
            "status": {
                "code": 200,
                "message": "Success",
            },
            "data": {
                "prediction": formatted_predictions,
            }
        }), 200
        
    else:
        return jsonify({
            "status": {
                "code": 405,
                "message": "Method not allowed"
            },
            "data": None,
        }), 405

if __name__ == "__main__":
    app.run()