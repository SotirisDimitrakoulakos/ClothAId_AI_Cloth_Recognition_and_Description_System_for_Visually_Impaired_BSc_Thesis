from flask import Flask, request, jsonify
from flask_ngrok import run_with_ngrok
import numpy as np
from PIL import Image
import io
import os
from prediction import ClothingAttributePredictor
from mmfashion_integration import MMFashionPredictor

app = Flask(__name__)
run_with_ngrok(app)  # This will expose your Colab server to the internet

# Initialize models
effnet_model, effnet_encoders = None, None  # Load these from your saved models
googlenet_model, googlenet_encoders = None, None  # Load these from your saved models
mmfashion_predictor = MMFashionPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400
    
    # Get image file
    image_file = request.files['image']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes))
    
    # Save temporarily (or process directly)
    temp_path = 'temp_image.jpg'
    image.save(temp_path)
    
    # Get model type if specified (default to EfficientNet)
    model_type = request.form.get('model_type', 'efficientnet')
    
    # Predict with selected model
    if model_type == 'efficientnet':
        predictor = ClothingAttributePredictor(effnet_model, effnet_encoders)
    else:
        predictor = ClothingAttributePredictor(googlenet_model, googlenet_encoders)
    
    # Get basic attributes
    attributes = predictor.predict_attributes(temp_path)
    
    # Get additional attributes from MMFashion based on articleType
    if 'articleType' in attributes:
        additional_attrs = mmfashion_predictor.predict(temp_path, attributes['articleType'])
        attributes.update(additional_attrs)
    
    # Clean up
    os.remove(temp_path)
    
    return jsonify(attributes)

if __name__ == '__main__':
    # Load models here
    # effnet_model, effnet_encoders = load_models(...)
    # googlenet_model, googlenet_encoders = load_models(...)
    
    app.run()