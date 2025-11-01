from flask import Flask, render_template, request, redirect, url_for, send_from_directory, jsonify
import os
from werkzeug.utils import secure_filename
import json
# from google import genai
import onnxruntime as ort
import numpy as np
import matplotlib.pyplot as plt
import cv2
import base64
import io
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
classesr = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy','Cherry_(including_sour)___Powdery_mildew','Cherry_(including_sour)___healthy','Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot','Corn_(maize)___Common_rust_','Corn_(maize)___Northern_Leaf_Blight','Corn_(maize)___healthy','Grape___Black_rot','Grape___Esca_(Black_Measles)','Grape___Leaf_blight_(Isariopsis_Leaf_Spot)','Grape___healthy','Orange___Huanglongbing_(Citrus_greening)','Peach___Bacterial_spot','Peach___healthy','Pepper,_bell___Bacterial_spot','Pepper,_bell___healthy','Potato___Early_blight','Potato___Late_blight','Potato___healthy','Raspberry___healthy','Soybean___healthy','Squash___Powdery_mildew','Strawberry___Leaf_scorch','Strawberry___healthy','Tomato___Bacterial_spot','Tomato___Early_blight','Tomato___Late_blight','Tomato___Leaf_Mold','Tomato___Septoria_leaf_spot','Tomato___Spider_mites Two-spotted_spider_mite','Tomato___Target_Spot','Tomato___Tomato_Yellow_Leaf_Curl_Virus','Tomato___Tomato_mosaic_virus','Tomato___healthy']
print("Loading ONNX model. This may take a moment...")

model_path = "static/best.onnx"
try:
    onnx_model = ort.InferenceSession(model_path, providers=['CUDAExecutionProvider'])
    print("Model loaded successfully.")
except Exception as e:
    print(f"FATAL ERROR: Could not load ONNX model from {model_path}")
    print(e)
        
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

def encode_image_to_base64(img_array):
    _, img_buffer = cv2.imencode('.jpg', img_array)
    return base64.b64encode(img_buffer).decode('utf-8')
    
def run_inference_on_image(image):
    annotated_frame = image.copy()

    # pre-processing
    img_w, img_h = image.shape[1], image.shape[0]
    img = cv2.resize(annotated_frame, (416, 416))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.transpose(2, 0, 1)
    img = img.reshape(1, 3, 416, 416)
    img = img / 255.0
    img = img.astype(np.float32)

    # inference
    outputs = onnx_model.run(None, {"images": img})

    # post-processing
    results = outputs[0]
    results = results.transpose()
    results = filter_Detections(results)

    if len(results) == 0:
        print("No detections found after filtering.")
        annotate_base64 = encode_image_to_base64(annotated_frame)
        return annotate_base64, []
    
    rescaled_results, confidences = rescale_back(results, img_w, img_h)

    all_detections_list = []
    # results = model.predict(os.path.join(path, "base.jpg"))
    # img = cv2.imread(os.path.join(path, "base.jpg"))
    # annotated_frame = results[0].plot()

    # loop through predictions
    for res, conf in zip(rescaled_results, confidences):
        x1,y1,x2,y2, cls_id = res
        cls_id = int(cls_id)
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        # draw on annotated frame
        conf_str = "{:.2f}".format(float(conf))
        class_name_raw = classesr[cls_id]
        pretty_class_name = classes.get(class_name_raw, class_name_raw)
        
        # draw the bounding boxes
        cv2.rectangle(annotated_frame,(int(x1),int(y1)),(int(x2),int(y2)),(255,0, 0),1)
        cv2.putText(annotated_frame, classesr[cls_id]+' '+conf_str,(x1,y1-17),
                    cv2.FONT_HERSHEY_SIMPLEX,0.7,(255,0,0),1)
        # get crop and encode
        y1, y2 = max(0, y1), min(img_h, y2)
        x1, x2 = max(0, x1), min(img_w, x2)
        screenshot = image[y1:y2, x1:x2]

        if screenshot.size == 0:
            continue
        screenshot = cv2.resize(screenshot, (256, 256))
        crop_base64 = encode_image_to_base64(screenshot)
        # Store detection data
        detection_data = {
            "bounding_box": [x1, y1, x2, y2],
            # "class_name": classes[model.names[cls_id]],
            "class name": pretty_class_name,
            "confidence": float(conf),
            "crop_image": crop_base64
        }
        all_detections_list.append(detection_data)
    annotated_image_base64 = encode_image_to_base64(annotated_frame)
    return annotated_image_base64, all_detections_list

@app.route("/infer", methods=["POST"])
def infer_image():
    if 'image' not in request.files:
        return jsonify({'error': 'no image procided'}), 400

    file_storage = request.files['image']

    # read the image in-memory
    in_memory_file = np.frombuffer(file_storage.read(), np.uint8)
    img = cv2.imdecode(in_memory_file, cv2.IMREAD_COLOR)

    if img is None:
        return jsonify({'error': 'invalid image format'}), 400

    print(f"Received image: {img.shape}")

    # run full inference pipeline
    try:
        annotated_base64, detections_list = run_inference_on_image(img)
    except Exception as e:
        print(f"An error ocurred during inference: {e}")
        return jsonify({'error': f'Inference failed: {str(e)}'}), 500

    response_data = {
        'annotated_image': annotated_base64,
        'all_detections': detections_list
    }

    print(f"Returning {len(detections_list)} detections.")
    return jsonify(response_data)
    
if __name__ == '__main__':
    print("Starting Flask server...")
    app.run(host = '0.0.0.0', port = 5000, debug=True)