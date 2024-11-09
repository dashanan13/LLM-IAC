import os
from flask import render_template, request, jsonify
from app import app
from app.detection import ShapeTextDetector
from werkzeug.utils import secure_filename
import cv2
import numpy as np
from PIL import Image
import traceback
import sys

detector = ShapeTextDetector()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'png', 'jpg', 'jpeg'}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    try:
        app.logger.info("Starting file upload process")
        
        if 'file' not in request.files:
            app.logger.error("No file part in request")
            return jsonify({'error': 'No file part'}), 400
        
        file = request.files['file']
        if file.filename == '':
            app.logger.error("No selected file")
            return jsonify({'error': 'No selected file'}), 400
            
        if file and allowed_file(file.filename):
            try:
                # Save original file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                app.logger.info(f"Saving file to: {filepath}")
                file.save(filepath)
                
                app.logger.info("File saved successfully, starting shape detection")
                
                # Process image
                processed_img, shapes, free_text = detector.detect_shapes_and_text(filepath)
                
                app.logger.info(f"Shape detection complete. Found {len(shapes)} shapes")
                
                # Save processed image
                processed_filename = 'processed_' + filename
                processed_filepath = os.path.join(app.config['PROCESSED_FOLDER'], processed_filename)
                app.logger.info(f"Saving processed image to: {processed_filepath}")
                
                # Ensure processed image is RGB
                if processed_img.mode != 'RGB':
                    processed_img = processed_img.convert('RGB')
                
                processed_img.save(processed_filepath, format='PNG')
                
                app.logger.info("Successfully saved processed image")
                
                # Print shapes for debugging
                app.logger.info(f"Detected shapes: {shapes}")
                
                # Convert numpy arrays to lists for JSON serialization
                shapes_json = []
                for shape in shapes:
                    shape_dict = {
                        'id': shape['id'],
                        'type': shape['type'],
                        'text': shape['text'],
                        'bounds': tuple(map(int, shape['bounds'])),
                        'center': tuple(map(int, shape['center']))
                    }
                    shapes_json.append(shape_dict)
                
                response_data = {
                    'original_image': f'/static/uploads/{filename}',
                    'processed_image': f'/static/processed/{processed_filename}',
                    'shapes': shapes_json,
                    'free_text': free_text
                }
                
                app.logger.info("Sending successful response")
                return jsonify(response_data)
                
            except Exception as e:
                app.logger.error(f"Error during processing: {str(e)}")
                app.logger.error(traceback.format_exc())
                return jsonify({'error': str(e)}), 500
        else:
            app.logger.error(f"Invalid file type: {file.filename}")
            return jsonify({'error': 'Invalid file type'}), 400
            
    except Exception as e:
        app.logger.error(f"Unexpected error: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500

@app.route('/add_shape', methods=['POST'])
def add_shape():
    try:
        app.logger.info("Starting add_shape process")
        
        data = request.json
        app.logger.info(f"Received data: {data}")
        
        required_fields = ['filename', 'shape_type', 'text', 'ref_shape1_id']
        missing_fields = [field for field in required_fields if field not in data]
        
        if missing_fields:
            error_msg = f"Missing required fields: {', '.join(missing_fields)}"
            app.logger.error(error_msg)
            return jsonify({'error': error_msg}), 400
        
        filename = data['filename']
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        
        if not os.path.exists(filepath):
            error_msg = f"Image file not found: {filepath}"
            app.logger.error(error_msg)
            return jsonify({'error': error_msg}), 404
        
        app.logger.info("Adding new shape to image")
        
        # Add new shape
        new_image, new_shape_id = detector.add_shape(
            filepath,
            data['shape_type'],
            data['text'],
            data['ref_shape1_id'],
            data.get('ref_shape2_id'),
            data.get('position', 'right')  # Default to 'right' if position is None
        )
        
        app.logger.info(f"Shape added successfully, ID: {new_shape_id}")
        
        # Save new image
        new_filename = 'new_' + filename
        new_filepath = os.path.join(app.config['PROCESSED_FOLDER'], new_filename)
        
        # Ensure image is RGB
        if new_image.mode != 'RGB':
            new_image = new_image.convert('RGB')
            
        new_image.save(new_filepath, format='PNG')
        
        app.logger.info("Successfully saved new image")
        
        return jsonify({
            'new_image': f'/static/processed/{new_filename}',
            'new_shape_id': new_shape_id
        })
        
    except Exception as e:
        app.logger.error(f"Error in add_shape: {str(e)}")
        app.logger.error(traceback.format_exc())
        return jsonify({'error': str(e)}), 500