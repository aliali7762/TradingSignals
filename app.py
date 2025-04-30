from flask import Flask, render_template, request, redirect, url_for, flash
import os
import cv2
import numpy as np
import pytesseract
from ultralytics import YOLO

app = Flask(__name__)
app.secret_key = 'your_secret_key'  # Replace with a real secret key in production

# Configure upload folder
UPLOAD_FOLDER = 'static/uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Ensure the upload folder exists
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Initialize the YOLOv8 model (uses a pre-trained model for demonstration)
model = YOLO('yolov8n.pt')  # The YOLO model file should be in your project directory

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Check if an image file was uploaded
        if 'chart_image' not in request.files:
            flash('No file part')
            return redirect(request.url)
        file = request.files['chart_image']
        if file.filename == '':
            flash('No selected file')
            return redirect(request.url)
        if file:
            # Save the uploaded image to the static/uploads folder
            filename = file.filename
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)

            # Perform YOLO object detection on the image
            results = model(filepath)[0]  # Run YOLOv8 inference
            detected_labels = []
            for box in results.boxes:
                class_id = int(box.cls)
                label = model.names[class_id]
                detected_labels.append(f"{label} ({box.conf:.2f})")

            # Perform OCR using pytesseract to extract text from the chart image
            # Ensure Tesseract OCR is installed on the system
            image = cv2.imread(filepath)
            text = pytesseract.image_to_string(image)

            # Simple example analysis (placeholder logic)
            trend = "Upward"  # Replace with real analysis logic

            # Render results page
            return render_template('result.html',
                                   filename=filename,
                                   detected_labels=detected_labels,
                                   ocr_text=text,
                                   trend=trend)

    # GET request - render upload form
    return render_template('index.html')

# Route to serve uploaded images
@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return redirect(url_for('static', filename='uploads/' + filename), code=301)

if __name__ == '__main__':
    app.run(debug=True)
