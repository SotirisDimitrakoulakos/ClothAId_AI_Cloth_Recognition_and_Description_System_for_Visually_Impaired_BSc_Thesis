from flask import Flask, request, jsonify
import io
from PIL import Image

# Import TensorFlow & your custom model classes
import tensorflow as tf
import pickle
import pandas as pd

from prediction import ClothingAttributePredictor
from mmfashion_integration import MMFashionPredictor

# Import your custom model builders and loaders here
from efficientnet_classifier import EfficientNetClothingClassifier
from iv3_classifier import IV3ClothingClassifier
from training_utils import ClothingClassifierTrainer

app = Flask(__name__)

balanced_metadata = pd.read_pickle('/content/drive/MyDrive/AI_Infrastructure/Data/pickle_pre_data/filtered_balanced_metadata_efnet.pkl')

# Define your classes count dict here or import it
num_classes_dict = {
    attr: len(balanced_metadata[attr].unique())
    for attr in ['masterCategory', 'subCategory', 'articleType',
                'baseColour', 'gender', 'season', 'usage']
}

def load_models():
    effnet = EfficientNetClothingClassifier(num_classes_dict)
    effnet_model = effnet.build_model()
    effnet_model, effnet_encoders = ClothingClassifierTrainer.load_model(
        'efficientnet', effnet_model, 'saved_models/EffNet_Fashion')

    iv3net = IV3ClothingClassifier(num_classes_dict)
    iv3net_model = iv3net.build_model()
    iv3net_model, iv3net_encoders = ClothingClassifierTrainer.load_model(
        'inceptionv3', iv3net_model, 'saved_models/IV3Net_Fashion')

    return effnet_model, effnet_encoders, iv3net_model, iv3net_encoders

effnet_model, effnet_encoders, iv3net_model, iv3net_encoders = load_models()
mmfashion_predictor = MMFashionPredictor()

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    image_file = request.files['image']
    image_bytes = image_file.read()
    image = Image.open(io.BytesIO(image_bytes)).convert('RGB')

    model_type = request.form.get('model_type', 'efficientnet')
    if model_type == 'efficientnet':
        predictor = ClothingAttributePredictor(effnet_model, effnet_encoders)
    else:
        predictor = ClothingAttributePredictor(iv3net_model, iv3net_encoders)

    attributes = predictor.predict_attributes_from_pil(image)

    if 'articleType' in attributes:
        additional_attrs = mmfashion_predictor.predict(image, attributes['articleType'])
        attributes.update(additional_attrs)

    return jsonify(attributes)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
