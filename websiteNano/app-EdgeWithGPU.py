from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
from werkzeug.utils import secure_filename
import json
# imports onnxruntime gpu
import onnxruntime as ort
import numpy as np
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

mode_path = "static/best.onnx"
# starts inference session with ONNX runtime, with its CUDA Execution Provider (GPU)
onnx_model = ort.InferenceSession(mode_path, providers=['CUDAExecutionProvider'])

# loads predictions.json file
def load_predictions():
    with open(PREDICTIONS_FOLDER, 'r') as f:
        return json.load(f)

# saves predictions
def save_predictions(entry):
    predictions = load_predictions()
    predictions.append(entry)
    with open(PREDICTIONS_FOLDER, 'w') as f:
        if predictions and predictions[-1] == entry:
            ordered = [entry] + predictions[:-1]
        else:
            ordered = [entry] + predictions
        json.dump(ordered, f)

def filter_Detections(results, thresh = 0.5):
    # if model is trained on 1 class only
    if len(results[0]) == 5:
        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in results if detection[4] > thresh]
        considerable_detections = np.array(considerable_detections)
        return considerable_detections

    # if model is trained on multiple classes
    else:
        A = []
        for detection in results:

            class_id = detection[4:].argmax()
            confidence_score = detection[4:].max()

            new_detection = np.append(detection[:4],[class_id,confidence_score])

            A.append(new_detection)

        A = np.array(A)

        # filter out the detections with confidence > thresh
        considerable_detections = [detection for detection in A if detection[-1] > thresh]
        considerable_detections = np.array(considerable_detections)

        return considerable_detections
        
def NMS(boxes, conf_scores, iou_thresh = 0.55):

    #  boxes [[x1,y1, x2,y2], [x1,y1, x2,y2], ...]

    x1 = boxes[:,0]
    y1 = boxes[:,1]
    x2 = boxes[:,2]
    y2 = boxes[:,3]

    areas = (x2-x1)*(y2-y1)

    order = conf_scores.argsort()

    keep = []
    keep_confidences = []

    while len(order) > 0:
        idx = order[-1]
        A = boxes[idx]
        conf = conf_scores[idx]

        order = order[:-1]

        xx1 = np.take(x1, indices= order)
        yy1 = np.take(y1, indices= order)
        xx2 = np.take(x2, indices= order)
        yy2 = np.take(y2, indices= order)

        keep.append(A)
        keep_confidences.append(conf)

        # iou = inter/union

        xx1 = np.maximum(x1[idx], xx1)
        yy1 = np.maximum(y1[idx], yy1)
        xx2 = np.minimum(x2[idx], xx2)
        yy2 = np.minimum(y2[idx], yy2)

        w = np.maximum(xx2-xx1, 0)
        h = np.maximum(yy2-yy1, 0)

        intersection = w*h

        # union = areaA + other_areas - intesection
        other_areas = np.take(areas, indices= order)
        union = areas[idx] + other_areas - intersection

        iou = intersection/union

        boleans = iou < iou_thresh

        order = order[boleans]

        # order = [2,0,1]  boleans = [True, False, True]
        # order = [2,1]

    return keep, keep_confidences



# function to rescale bounding boxes 
def rescale_back(results,img_w,img_h):
    cx, cy, w, h, class_id, confidence = results[:,0], results[:,1], results[:,2], results[:,3], results[:,4], results[:,-1]
    cx = cx/416.0 * img_w
    cy = cy/416.0 * img_h
    w = w/416.0 * img_w
    h = h/416.0 * img_h
    x1 = cx - w/2
    y1 = cy - h/2
    x2 = cx + w/2
    y2 = cy + h/2

    boxes = np.column_stack((x1, y1, x2, y2, class_id))
    keep, keep_confidences = NMS(boxes,confidence)
    print(np.array(keep).shape)
    return keep, keep_confidences

# running inference with GPU
def predict_model(index, path):
    # reads image
    image = cv2.imread(os.path.join(path, "base.jpg"))
    annotated_frame = image.copy()

    # assign variables
    img_w, img_h = image.shape[1], image.shape[0]

    # image preprocessing
    img = cv2.resize(image, (416, 416))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, 3, 416, 416)
    # normalising
    img = img / 255.0
    img = img.astype(np.float32)

    # runs inference on onnx runtime gpu
    outputs = onnx_model.run(None, {"images": img})
    results = outputs[0]
    results = results.transpose()
    # filter detections
    results = filter_Detections(results)
    # scale it to correct format
    rescaled_results, confidences = rescale_back(results, img_w, img_h)
    
    # save to predictions
    save_path = os.path.join(path, "predictions")
    # classes list directly linked with dataset
    classesr = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Huanglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']
    
    # loop through detections
    # to draw annotated image
    for res, conf in zip(rescaled_results, confidences):
        x1,y1,x2,y2, cls_id = res
        cls_id = int(cls_id)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        conf = "{:.2f}".format(conf)
        # draw the bounding boxes
        cv2.rectangle(annotated_frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0, 0),1)
        cv2.putText(annotated_frame, classesr[cls_id]+' '+conf,(x1,y1-17),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),1)
    

    # save annotated image
    cv2.imwrite(os.path.join(path, "annotated.jpg"), annotated_frame)
    data = {
        "file_url": f"/static/predictions/folder_{index}/annotated.jpg",
        "file_index": index,
        "predictions": {}
    }

    # extract 3 most confident predictions
    three_most_confident_predictions = []
    three_most_confident_values = []
    three_most_confident_files = []
    image = cv2.imread(os.path.join(path, "base.jpg"))
    i = 0

    # loop through detections
    for res, conf in zip(rescaled_results, confidences):
        x1,y1,x2,y2, cls_id = res
        cls_id = int(cls_id)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        screenshot = image[y1:y2, x1:x2]
        print('here', screenshot.shape)
        crop_path = os.path.join(path, f"crop_{i}.jpg")

        # crop original image into bounding boxes
        screenshot = cv2.resize(screenshot, (256, 256))
        cv2.imwrite(crop_path, screenshot)
        crop_url = f"/static/predictions/folder_{index}/crop_{i}.jpg"
        data["predictions"][f"crop_{i}"] = {
            "bounding_box": [x1, y1, x2, y2],
            "class_id": cls_id,
            # "class_name": classes[model.names[cls_id]],
            "class name": classesr[cls_id],
            "confidence": conf,
            "crop_url": crop_url
        }

        # add to three_most_confident_predictions
        if len(three_most_confident_predictions) < 3:
            three_most_confident_predictions.append(classesr[cls_id])
            three_most_confident_values.append(conf)
            three_most_confident_files.append(crop_url)
        else:
            min_conf_index = three_most_confident_values.index(min(three_most_confident_values))
            if conf > three_most_confident_values[min_conf_index]:
                three_most_confident_predictions[min_conf_index] = classesr[cls_id]
                three_most_confident_values[min_conf_index] = conf
                three_most_confident_files[min_conf_index] = crop_url
        i+=1
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
