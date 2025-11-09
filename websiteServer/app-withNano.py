from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
from werkzeug.utils import secure_filename
import json
import matplotlib.pyplot as plt
import cv2
from openai import OpenAI
import datetime
# necessary libraries for NANO
import requests
import base64 
import numpy as np
import io  

NANO_API_URL = "" # Replace with Nano IP/infer

classes = {
    "Apple___Apple_scab": "Apple Scab",
    "Apple___Black_rot": "Black Rot",
    "Apple___Cedar_apple_rust": "Cedar Apple Rust",
    "Apple___healthy": "Healthy Apple",
    "Blueberry___healthy": "Healthy Blueberry",
    "Cherry_(including_sour)___Powdery_mildew": "Powdery Mildew",
    "Cherry_(including_sour)___healthy": "Healthy Cherry",
    "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot": "Gray Leaf Spot",
    "Corn_(maize)___Common_rust_": "Common Rust",
    "Corn_(maize)___Northern_Leaf_Blight": "Northern Leaf Blight",
    "Corn_(maize)___healthy": "Healthy Corn",
    "Grape___Black_rot": "Black Rot",
    "Grape___Esca_(Black_Measles)": "Esca (Black Measles)",
    "Grape___Leaf_blight_(Isariopsis_Leaf_Spot)": "Leaf Blight (Isariopsis Leaf Spot)",
    "Grape___healthy": "Healthy Grape",
    "Orange___Haunglongbing_(Citrus_greening)": "Huanglongbing (Citrus greening)",
    "Peach___Bacterial_spot": "Bacterial Spot",
    "Peach___healthy": "Healthy Peach",
    "Pepper,_bell___Bacterial_spot": "Bacterial Spot",
    "Pepper,_bell___healthy": "Healthy Pepper Bell",
    "Potato___Early_blight": "Early Blight",
    "Potato___Late_blight": "Late Blight",
    "Potato___healthy": "Healthy Potato",
    "Raspberry___healthy": "Healthy Raspberry",
    "Soybean___healthy": "Healthy Soybean",
    "Squash___Powdery_mildew": "Powdery Mildew",
    "Strawberry___Leaf_scorch": "Leaf Scorch",
    "Strawberry___healthy": "Healthy Strawberry",
    "Tomato___Bacterial_spot": "Bacterial Spot",
    "Tomato___Early_blight": "Early Blight",
    "Tomato___Late_blight": "Late Blight",
    "Tomato___Leaf_Mold": "Leaf Mold",
    "Tomato___Septoria_leaf_spot": "Septoria Leaf Spot",
    "Tomato___Spider_mites Two-spotted_spider_mite": "Two-spotted Spider Mite",
    "Tomato___Target_Spot": "Target Spot",
    "Tomato___Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato___healthy": "Healthy Tomato"
}
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
STATIC_FOLDER = os.path.join(app.root_path, 'static')
STATIC_PREDICTIONS_FOLDER = os.path.join(STATIC_FOLDER, "predictions")
PREDICTIONS_FOLDER = os.path.join(app.root_path, 'data/predictions.json')

# model = YOLO('website/static/best.pt') <-- REMOVED (No longer needed)

def load_predictions():
    with open(PREDICTIONS_FOLDER, 'r') as f:
        return json.load(f)
    
def save_predictions(entry):
    predictions = load_predictions()
    predictions.append(entry)
    with open(PREDICTIONS_FOLDER, 'w') as f:
        if predictions and predictions[-1] == entry:
            ordered = [entry] + predictions[:-1]
        else:
            ordered = [entry] + predictions
        json.dump(ordered, f)

# --- vvv THIS IS THE MODIFIED FUNCTION vvv ---

def decode_and_save_image(base64_string, save_path):
    """Helper function to decode Base64 string and save as an image file."""
    try:
        # Decode the Base64 string
        img_data = base64.b64decode(base64_string)
        
        # Convert binary data to a numpy array
        nparr = np.frombuffer(img_data, np.uint8)
        
        # Decode image
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Could not decode image")
            
        # Save the image
        cv2.imwrite(save_path, img)
        return True
    except Exception as e:
        print(f"Error decoding/saving image at {save_path}: {e}")
        return False

def predict_model(index, path):
    """
    Sends the image to the Jetson Nano for inference and processes the results.
    """
    base_image_path = os.path.join(path, "base.jpg")
    
    # 1. Prepare the image file to send to the Nano
    try:
        with open(base_image_path, 'rb') as f:
            files_to_send = {
                'image': (os.path.basename(base_image_path), f.read(), 'image/jpeg')
            }
    except FileNotFoundError:
        print(f"Error: Could not find file {base_image_path}")
        return [], [], [], 0, []

    # 2. Send the request to the Jetson Nano
    try:
        print(f"Sending request to Nano at {NANO_API_URL}...")
        response = requests.post(NANO_API_URL, files=files_to_send, timeout=30) # 30s timeout
        response.raise_for_status() # Raise an exception for bad status codes (4xx, 5xx)
        
        # Get the JSON response from the Nano
        nano_data = response.json()
        print("Received response from Nano.")

    except requests.exceptions.ConnectionError:
        print("Error: Could not connect to the Jetson Nano.")
        return [], [], [], 0, []
    except requests.exceptions.Timeout:
        print("Error: Inference request to Nano timed out.")
        return [], [], [], 0, []
    except requests.exceptions.RequestException as e:
        print(f"Error sending request to Nano: {e}")
        return [], [], [], 0, []
    except json.JSONDecodeError:
        print("Error: Could not decode JSON response from Nano.")
        return [], [], [], 0, []

    # 3. Process the Nano's response (this must match your Nano's API output)
    
    # This structure assumes your Nano sends back a JSON like:
    # {
    #   "annotated_image": "...",  # Base64 string
    #   "all_detections": [
    #     {
    #       "bounding_box": [x1, y1, x2, y2],
    #       "class_name": "Apple___Apple_scab",
    #       "confidence": 0.95,
    #       "crop_image": "..." # Base64 string
    #     },
    #     ...
    #   ]
    # }

    if 'annotated_image' not in nano_data or 'all_detections' not in nano_data:
        print("Error: Nano response is missing required keys.")
        return [], [], [], 0, []

    # 4. Save the annotated image
    annotated_save_path = os.path.join(path, "annotated.jpg")
    decode_and_save_image(nano_data['annotated_image'], annotated_save_path)

    # 5. Process and save all detections and crops
    data = {
        "file_url": f"/static/predictions/folder_{index}/annotated.jpg",
        "file_index": index,
        "predictions": {}
    }
    three_most_confident_predictions = []
    three_most_confident_values = []
    three_most_confident_files = []
    types_of_diseases = []

    detections = nano_data['all_detections']    
    # Sort detections by confidence (highest first)
    detections.sort(key=lambda x: x['confidence'], reverse=True)

    for i, det in enumerate(detections):
        crop_path = os.path.join(path, f"crop_{i}.jpg")
        crop_url = f"/static/predictions/folder_{index}/crop_{i}.jpg"
        
        # Decode and save the crop image sent from the Nano
        decode_and_save_image(det['crop_image'], crop_path)
        
        # Get data from the Nano's response
        x1, y1, x2, y2 = det['bounding_box']
        conf = float(det['confidence'])
        class_name = det['class name'] # Assumes Nano sends the pretty name

        data["predictions"][f"crop_{i}"] = {
            "bounding_box": [x1, y1, x2, y2],
            "class_id": "N/A (from Nano)", # Or get class_id from Nano if available
            "class_name": class_name,
            "confidence": conf,
            "crop_url": crop_url
        }
        types_of_diseases.append(class_name)
        # Populate the top 3 predictions
        if i < 3:
            three_most_confident_predictions.append(class_name)
            three_most_confident_values.append(conf)
            three_most_confident_files.append(crop_url)

    # Save the data.json file
    with open(os.path.join(path, "data.json"), 'w') as f:
        json.dump(data, f)

    # Ensure the lists have 3 items for consistency with your old code
    while len(three_most_confident_predictions) < 3:
        three_most_confident_predictions.append("None")
    while len(three_most_confident_files) < 3:
        three_most_confident_files.append("")
    while len(three_most_confident_values) < 3:
        three_most_confident_values.append(0.0)

    return three_most_confident_predictions, three_most_confident_files, three_most_confident_values, len(detections), types_of_diseases

# --- ^^^ THIS IS THE MODIFIED FUNCTION ^^^ ---


def generate_response(result):
    if "Healthy" in result:
        return "Your plant is healthy! Keep up the good work!"
    else:
        content = f"My plant has {result}. Present some solutions in point form, under 60 words long. Write your response and suggestions not in markdown but in HTML surrounded by a <p>."
        client = OpenAI(api_key="sk-00f4b5e7e8514c3797af6e8db63b189b", base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": content},
        ],
            max_tokens=1024,
            temperature=0.7,
            stream=False
        )
        return response.choices[0].message.content
    
# Redirect to /index from root (server initializes on /)
@app.route("/")
def home():
    return redirect(url_for('index'))

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(UPLOAD_FOLDER, filename)

@app.route("/analyze/<pred>", methods=['GET', 'POST'])
def predict(pred):
    return generate_response(pred)

@app.route("/about")
def about():
    return render_template('about.html')

@app.route('/save', methods=['POST'])
def save_json():
    data = request.get_json(force=True)
    try:
        datumIndex = int(data.get('datumIndex'))
        pIndex = int(data.get('pIndex'))
        pred = data.get('pred', '')
    except Exception:
        return jsonify({'status': 'bad_request'}), 400

    predictions = load_predictions()
    if 0 <= datumIndex < len(predictions):
        analyses = predictions[datumIndex].get("analyses", ["", "", ""])
        if 0 <= pIndex < len(analyses):
            analyses[pIndex] = pred
            predictions[datumIndex]["analyses"] = analyses
            with open(PREDICTIONS_FOLDER, 'w') as f:
                json.dump(predictions, f)
            return jsonify({'status': 'ok'})
    return jsonify({'status': 'not_found'}), 404

@app.route('/predict', methods=['GET', 'POST'])
def index():
    prediction = None
    file_url = None
    if request.method == 'POST':
        if 'img' in request.files:
            file = request.files['img']
            if file and file.filename:
                predictions = load_predictions()
                predictions_length = len(predictions)
                filename = secure_filename(file.filename)
                folder_path = os.path.join(STATIC_PREDICTIONS_FOLDER, f"folder_{predictions_length}")
                os.makedirs(folder_path, exist_ok=True)
                
                # Save the base image
                base_save_path = os.path.join(folder_path, "base.jpg")
                file.save(base_save_path)
                
                # Reset file pointer to save to UPLOAD_FOLDER as well
                file.stream.seek(0)
                upload_save_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(upload_save_path)
                
                file_url = f"/static/predictions/folder_{predictions_length}/base.jpg"
                
                # *** This function now calls the Nano ***
                prediction, crops, confidences, boxes_quan, types = predict_model(predictions_length, folder_path)
                
                # Check if predict_model failed (e.g., Nano connection error)
                if not prediction:
                    # Handle the error, maybe render an error message
                    print("Prediction failed, likely a Nano communication issue.")
                    # You might want to pass an error to the template
                    return render_template('predict.html', data=load_predictions(), quan=int(request.args.get("quan", 1)), classes = classes, error="Failed to process image. Could not connect to inference server.")

                save_predictions({'file_url': file_url,
                                    "annotated_url": f"/static/predictions/folder_{predictions_length}/annotated.jpg",
                                    "prediction": prediction,
                                    "index": len(predictions),
                                    "prediction_data": [
                                        {
                                            "class_name": prediction[i],
                                            "confidence": round(confidences[i], 3),
                                            "crop_url": crops[i]
                                        } for i in range(3)
                                    ],
                                    "analyses": ["", "", ""],
                                    "boxes_quan": boxes_quan,
                                    "types": types
                                })
    return render_template('predict.html', data=load_predictions(), quan=int(request.args.get("quan", 1)), classes = classes)

if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=8000) # Recommend specifying host and port
