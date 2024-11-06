from flask import Flask, request, jsonify, render_template
import cv2
import numpy as np
from PIL import Image
import pytesseract
import io
import os
import uuid

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['UPLOAD_FOLDER'] = 'uploads'

if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

class ShapeInfo:
    def __init__(self, shape_type, text, contour):
        self.id = str(uuid.uuid4())[:8]
        self.shape_type = shape_type
        self.text = text.strip() if text else ""
        self.contour = contour

    def __str__(self):
        if self.text:
            return f"{self.shape_type} containing text: '{self.text}'"
        return f"{self.shape_type} (no text)"

def preprocess_for_ocr(image, contour):
    # Create a slightly larger mask to ensure we don't clip text
    x, y, w, h = cv2.boundingRect(contour)
    padding = 5
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv2.drawContours(mask, [contour], -1, (255), -1)
    
    # Apply mask to image
    result = cv2.bitwise_and(image, image, mask=mask)
    
    # Crop to ROI with padding
    roi = result[max(0, y-padding):min(result.shape[0], y+h+padding), 
                max(0, x-padding):min(result.shape[1], x+w+padding)]
    
    if roi.size == 0:
        return None

    # Convert to grayscale
    gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
    
    # Apply multiple preprocessing techniques
    # 1. Increase contrast
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    contrast = clahe.apply(gray)
    
    # 2. Remove noise
    denoised = cv2.fastNlMeansDenoising(contrast)
    
    # 3. Thresholding with different methods
    thresh1 = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    thresh2 = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                  cv2.THRESH_BINARY, 11, 2)
    
    # 4. Dilation to connect components
    kernel = np.ones((2,2), np.uint8)
    dilated = cv2.dilate(thresh1, kernel, iterations=1)
    
    return [thresh1, thresh2, dilated]

def extract_text_from_region(image, contour):
    # Preprocess image in multiple ways
    processed_images = preprocess_for_ocr(image, contour)
    if not processed_images:
        return ""
    
    best_text = ""
    max_confidence = 0
    
    # OCR configurations to try
    configs = [
        '--psm 6 --oem 3',  # Assume uniform block of text
        '--psm 7 --oem 3',  # Treat the image as a single text line
        '--psm 8 --oem 3',  # Treat the image as a single word
        '--psm 13 --oem 3'  # Raw line with default orientation
    ]
    
    for processed in processed_images:
        pil_image = Image.fromarray(processed)
        
        for config in configs:
            try:
                # Get detailed OCR results
                data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
                
                # Calculate average confidence for this result
                confidences = [int(conf) for conf in data['conf'] if conf != '-1']
                if confidences:
                    avg_confidence = sum(confidences) / len(confidences)
                    text = ' '.join([word for word in data['text'] if word.strip()])
                    
                    if text and avg_confidence > max_confidence:
                        max_confidence = avg_confidence
                        best_text = text
            except Exception as e:
                continue
    
    return best_text

def detect_shapes(image_path):
    # Read image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Could not read image")
    
    # Resize if image is too large
    max_dimension = 1500
    height, width = image.shape[:2]
    if max(height, width) > max_dimension:
        scale = max_dimension / max(height, width)
        image = cv2.resize(image, None, fx=scale, fy=scale)
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Enhanced preprocessing
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 50, 150)
    kernel = np.ones((3,3), np.uint8)
    dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Find contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    shapes = []
    min_area = 100
    
    # Debug image
    debug_image = image.copy()
    
    for contour in contours:
        if cv2.contourArea(contour) < min_area:
            continue
            
        # Approximate the contour
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.04 * peri, True)
        
        # Improved shape detection
        vertices = len(approx)
        area = cv2.contourArea(contour)
        circularity = 4 * np.pi * area / (peri * peri)
        
        x, y, w, h = cv2.boundingRect(approx)
        aspect_ratio = float(w)/h
        
        # More precise shape identification
        if circularity > 0.85:
            shape_type = "Circle"
        elif vertices == 3:
            shape_type = "Triangle"
        elif vertices == 4:
            if 0.95 <= aspect_ratio <= 1.05:
                shape_type = "Square"
            else:
                # Check for diamond
                angles = []
                for i in range(4):
                    pt1 = approx[i][0]
                    pt2 = approx[(i+1)%4][0]
                    pt3 = approx[(i+2)%4][0]
                    angle = abs(np.degrees(np.arctan2(pt3[1]-pt2[1], pt3[0]-pt2[0]) - 
                                        np.arctan2(pt1[1]-pt2[1], pt1[0]-pt2[0])))
                    angles.append(angle)
                if all(abs(angle - 90) < 15 for angle in angles):
                    if abs(approx[0][0][0] - approx[2][0][0]) < 10 or abs(approx[1][0][1] - approx[3][0][1]) < 10:
                        shape_type = "Diamond"
                    else:
                        shape_type = "Rectangle"
                else:
                    shape_type = "Rectangle"
        elif vertices == 6:
            shape_type = "Hexagon"
        else:
            shape_type = f"Polygon ({vertices} sides)"
        
        # Extract text from shape region
        text = extract_text_from_region(image, contour)
        
        # Create shape info object
        shape_info = ShapeInfo(shape_type, text, contour)
        shapes.append(shape_info)
        
        # Draw on debug image
        cv2.drawContours(debug_image, [approx], -1, (0, 255, 0), 2)
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            label = f"{shape_type}"
            if text:
                label += f": {text[:15]}..."
            cv2.putText(debug_image, label, (cX - 20, cY), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    
    # Save debug image
    debug_path = image_path.replace('.', '_debug.')
    cv2.imwrite(debug_path, debug_image)
    
    return shapes

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
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        
        try:
            shapes = detect_shapes(filepath)
            results = []
            for shape in shapes:
                shape_info = {
                    'id': shape.id,
                    'type': shape.shape_type,
                    'text': shape.text if shape.text else 'No text detected'
                }
                results.append(shape_info)
            
            return jsonify({
                'status': 'success',
                'shapes': results
            })
        except Exception as e:
            return jsonify({'error': str(e)}), 500
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
            debug_path = filepath.replace('.', '_debug.')
            if os.path.exists(debug_path):
                os.remove(debug_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)