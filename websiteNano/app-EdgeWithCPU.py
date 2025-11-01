from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
from werkzeug.utils import secure_filename
import json
# from google import genai
from ultralytics import YOLO
import matplotlib.pyplot as plt
import cv2
from openai import OpenAI

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
    "Tomato___Tomato_Yellow_Leaf_Curl_Virus": "Yellow Leaf Curl Virus",
    "Tomato___Tomato_mosaic_virus": "Tomato Mosaic Virus",
    "Tomato___healthy": "Healthy Tomato"
}
app = Flask(__name__)

UPLOAD_FOLDER = os.path.join(app.root_path, 'uploads')
STATIC_FOLDER = os.path.join(app.root_path, 'static')
STATIC_PREDICTIONS_FOLDER = os.path.join(STATIC_FOLDER, "predictions")
PREDICTIONS_FOLDER = os.path.join(app.root_path, 'data/predictions.json')

model = YOLO('static/best.pt')

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
        
def predict_model(index, path):
    results = model.predict(os.path.join(path, "base.jpg"))
    save_path = os.path.join(path, "predictions")
    img = cv2.imread(os.path.join(path, "base.jpg"))
    annotated_frame = results[0].plot()
    cv2.imwrite(os.path.join(path, "annotated.jpg"), annotated_frame)
    boxes = results[0].boxes
    data = {
        "file_url": f"/static/predictions/folder_{index}/annotated.jpg",
        "file_index": index,
        "predictions": {}
    }
    three_most_confident_predictions = []
    three_most_confident_values = []
    three_most_confident_files = []
    for i in range(len(boxes)):
        x1, y1, x2, y2 = map(int, boxes.xyxy[i])
        conf = float(boxes.conf[i])
        cls_id = int(boxes.cls[i])
        screenshot = img[y1:y2, x1:x2]
        crop_path = os.path.join(path, f"crop_{i}.jpg")
        screenshot = cv2.resize(screenshot, (256, 256))
        cv2.imwrite(crop_path, screenshot)
        crop_url = f"/static/predictions/folder_{index}/crop_{i}.jpg"
        print(model.names[cls_id])
        data["predictions"][f"crop_{i}"] = {
            "bounding_box": [x1, y1, x2, y2],
            "class_id": cls_id,
            # "class_name": classes[model.names[cls_id]],
            "class name": model.names[cls_id],
            "confidence": conf,
            "crop_url": crop_url
        }
        if len(three_most_confident_predictions) < 3:
            three_most_confident_predictions.append(classes[model.names[cls_id]])
            three_most_confident_values.append(conf)
            three_most_confident_files.append(crop_url)
        else:
            min_conf_index = three_most_confident_values.index(min(three_most_confident_values))
            if conf > three_most_confident_values[min_conf_index]:
                three_most_confident_predictions[min_conf_index] = classes[model.names[cls_id]]
                three_most_confident_values[min_conf_index] = conf
                three_most_confident_files[min_conf_index] = crop_url
    with open(os.path.join(path, "data.json"), 'w') as f:
        json.dump(data, f)
    while len(three_most_confident_predictions) < 3:
        three_most_confident_predictions.append("None")
    while len(three_most_confident_files) < 3:
        three_most_confident_files.append("")
    while len(three_most_confident_values) < 3:
        three_most_confident_values.append(0.0)
    return three_most_confident_predictions, three_most_confident_files, three_most_confident_values

def generate_response(result):
    if "Healthy" in result:
        content = f"My plant is {result}. How might I prepare for potential diseases?"
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
    return "Your plant is healthy! Keep up the good work!"

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
                save_path = os.path.join(folder_path, "base.jpg")
                file.save(save_path)
                file_url = f"/static/predictions/folder_{predictions_length}/base.jpg"
                save_path = os.path.join(UPLOAD_FOLDER, filename)
                file.save(save_path)
                prediction, crops, confidences = predict_model(predictions_length, folder_path)
                save_predictions({'file_url': file_url,
                                    "annotated_url": f"/static/predictions/folder_{predictions_length}/annotated.jpg",
                                    "prediction": prediction,
                                    "prediction_data": [
                                        {
                                            "class_name": prediction[i],
                                            "confidence": confidences[i],
                                            "crop_url": crops[i]
                                        } for i in range(3)
                                    ],
                                    "analyses": ["", "", ""]
                                })
    return render_template('predict.html', data=load_predictions(), quan=int(request.args.get("quan", 1)))

if __name__ == '__main__':
    app.run(debug=True)