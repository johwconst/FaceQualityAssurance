from flask import Flask, request, jsonify
from face_qa.face_qa import FaceQA
import base64
import cv2
import numpy as np
import tempfile
import os
import requests

app = Flask(__name__)

def analyze_image_from_base64(base64_str):
    if ',' in base64_str:
        base64_str = base64_str.split(',')[1]

    # Decode base64 to bytes
    image_data = base64.b64decode(base64_str)
    np_array = np.frombuffer(image_data, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Save the image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        temp_image_path = tmp_file.name
        cv2.imwrite(temp_image_path, image)

    print(f"Temporary image saved at: {temp_image_path}")
    # Run FaceQA
    face_qa = FaceQA(temp_image_path, 1)
    result = face_qa.check_face()

    # Clean up temp file
    os.remove(temp_image_path)

    return result

def analyze_image_from_url_download(url_image):
    # Download the image from the URL
    response = requests.get(url_image)
    if response.status_code != 200:
        raise Exception(f"Failed to download image from URL: {url_image}")

    # Decode the image data
    np_array = np.frombuffer(response.content, np.uint8)
    image = cv2.imdecode(np_array, cv2.IMREAD_COLOR)

    # Save the image temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        temp_image_path = tmp_file.name
        cv2.imwrite(temp_image_path, image)

    print(f"Temporary image saved at: {temp_image_path}")
    # Run FaceQA
    face_qa = FaceQA(temp_image_path, 1)
    result = face_qa.check_face()

    # Clean up temp file
    os.remove(temp_image_path)

    return result

def convert_np_types(obj):
    """Recursively convert numpy types to native Python types."""
    if isinstance(obj, dict):
        return {k: convert_np_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_np_types(v) for v in obj]
    elif isinstance(obj, np.generic):  # covers numpy.bool_, numpy.int32, etc
        return obj.item()
    else:
        return obj

@app.route('/base64', methods=['POST'])
def analyze_base64():
    data = request.get_json()

    if not data or 'image' not in data:
        return jsonify({"error": "Missing 'image' field in JSON"}), 400

    base64_image = data['image']

    try:
        result = convert_np_types(analyze_image_from_base64(base64_image))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route('/url', methods=['POST'])
def analyze_url():
    data = request.get_json()
    if not data or 'url' not in data:
        return jsonify({"error": "Missing 'url' field in JSON"}), 400

    url_image = data['url']

    if not url_image:
        return jsonify({"error": "Missing 'url' parameter"}), 400

    try:
        result = convert_np_types(analyze_image_from_url_download(url_image))
        return jsonify(result)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
