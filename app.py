from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from PIL import Image
from collections import Counter
import io
import os

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

def detect_shapes(image_path):
    # Read image
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur to reduce noise
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Apply adaptive thresholding
    threshold = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                    cv2.THRESH_BINARY_INV, 11, 2)
    
    # Find contours
    contours, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    min_area = 100  # Minimum area to consider (filters out noise)
    
    for contour in contours:
        # Filter out small contours
        if cv2.contourArea(contour) < min_area:
            continue
            
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.03 * peri, True)
        
        # Get number of vertices
        vertices = len(approx)
        
        # Calculate circularity
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (peri * peri)
        
        # Identify shape based on vertices and geometric properties
        if circularity > 0.85:  # More permissive circle detection
            shape = "Circle"
        elif vertices == 3:
            shape = "Triangle"
        elif vertices == 4:
            x, y, w, h = cv2.boundingRect(approx)
            aspect_ratio = float(w)/h
            if 0.95 <= aspect_ratio <= 1.05:
                shape = "Square"
            else:
                shape = "Rectangle"
        elif vertices == 6:
            shape = "Hexagon"
        else:
            shape = f"Polygon ({vertices} sides)"
            
        shapes.append(shape)
        
        # Debug: Draw contours and shape names on image
        cv2.drawContours(image, [approx], -1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            cv2.putText(image, shape, (cX - 20, cY), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (255, 0, 0), 2)
    
    # Save debug image
    debug_path = image_path.replace('.', '_debug.')
    cv2.imwrite(debug_path, image)
    
    # Count shapes
    shape_counts = Counter(shapes)
    return [f"{shape}: {count}" for shape, count in shape_counts.items()]

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
        
    if file:
        # Save the file
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        # Process the image
        try:
            shapes = detect_shapes(filepath)
            return jsonify({
                'status': 'success',
                'shapes': shapes
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            # Clean up
            if os.path.exists(filepath):
                os.remove(filepath)
            # Remove debug image
            debug_path = filepath.replace('.', '_debug.')
            if os.path.exists(debug_path):
                os.remove(debug_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
