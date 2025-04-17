import os
from flask import Flask, render_template, request, send_from_directory
import cv2
import pytesseract
import numpy as np


pytesseract.pytesseract.tesseract_cmd = "C:/Users/Shailendra Mandal/AppData/Local/Programs/Tesseract-OCR/tesseract.exe"

app = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
OUTPUT_FOLDER = 'output_path/ocr'

if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

if not os.path.exists(OUTPUT_FOLDER):
    os.makedirs(OUTPUT_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
def get_string(img_path):
    image = cv2.imread(img_path)

    file_name = os.path.basename(img_path).split('.')[0]
    file_name = file_name.split()[0]

    output_path = os.path.join(OUTPUT_FOLDER, file_name + "_filter_" + ".png")
    image = cv2.resize(image, None, fx=1.5, fy=1.5, interpolation=cv2.INTER_CUBIC)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    dilated = cv2.dilate(image, np.ones((7, 7), np.uint8))
    bg = cv2.medianBlur(dilated, 21)
    enhanced = 255 - cv2.absdiff(image, bg)

    kernel = np.array([[0, -1, 0], [-1, 5,-1], [0, -1, 0]])
    enhanced = cv2.filter2D(image, -1, kernel)
    
    image=enhanced

    kernel = np.ones((1, 1), np.uint8)
    image = cv2.dilate(image, kernel, iterations=1)
    image = cv2.erode(image, kernel, iterations=1)

    cv2.imwrite(output_path, image)

    result = pytesseract.image_to_string(image, lang="eng")
    return result


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/upload', methods=['POST'])
def upload_image():
    if 'file' not in request.files:
        return "No file part"
    
    file = request.files['file']
    if file.filename == '':
        return "No selected file"

    
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
    file.save(filepath)

    
    result = get_string(filepath)


    lines = result.split('\n')

    return render_template('result.html', lines=lines)


if __name__ == "__main__":
    app.run(debug=True)
