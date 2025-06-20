from flask import Flask, request, jsonify
import io
from PIL import Image
import logging
import traceback
import pandas as pd

from prediction import ClothingAttributePredictor

# Your model and training imports
from training import ClothingClassifierTrainer
from efficientnet_model import EfficientNetClothingClassifier
from articleType_efnet_model import EfficientNetArtTypeClothingClassifier

app = Flask(__name__)
logging.basicConfig(level=logging.INFO)

# Load balanced metadata (adjust path as needed)
balanced_metadata = pd.read_parquet('C:\\Users\\DimSot1\\Documents\\University\\BSc_THESIS\\ClothAId_AI_Cloth_Recognition_and_Description_System_for_Visually_Impaired_BSc_Thesis\\AI_infrastructure\\Data\\pre_data_2\\EFN\\filtered_balanced_dataset_ef_2.parquet')

num_classes_dict = {
    attr: len(balanced_metadata[attr].unique())
    for attr in ['masterCategory', 'subCategory', 'articleType',
                'baseColour', 'gender', 'season', 'usage']
}

def load_multi_task_model():
    logging.info("Loading multi-task model (EffNet_Fashion_3)...")
    multi_model_arch = EfficientNetClothingClassifier(num_classes_dict).build_model()
    multi_model_dir = '../Saved_Models/EffNet_Fashion_3/fine_tuned/final'
    multi_model, multi_label_encoders = ClothingClassifierTrainer.load_model(
        model_name='efficientnet',
        model_arch=multi_model_arch,
        save_dir=multi_model_dir,
        best_weights=False
    )
    return multi_model, multi_label_encoders

def load_single_task_model():
    logging.info("Loading single-task articleType model (EffNet_artTypeB1)...")
    single_model_arch = EfficientNetArtTypeClothingClassifier(87, 'efficientnetB1_artType', (240, 240, 3)).build_fine_tuned_model()
    single_model_dir = '../Saved_Models/EffNet_artTypeB1/final'
    single_model, single_label_encoders = ClothingClassifierTrainer.load_model(
        model_name='efficientnetB1_artType',
        model_arch=single_model_arch,
        save_dir=single_model_dir,
        best_weights=False,
        only_artType=True
    )
    return single_model, single_label_encoders

# Load models once on startup
multi_model, multi_label_encoders = load_multi_task_model()
single_model, single_label_encoders = load_single_task_model()

multi_predictor = ClothingAttributePredictor(multi_model, multi_label_encoders, 'EffNet_Fashion_3')
single_predictor = ClothingAttributePredictor(single_model, single_label_encoders, 'EffNet_artTypeB1')

@app.route('/', methods=['GET'])
def home():
    return "Flask server is running and reachable!"

@app.route('/predict', methods=['POST'])
def predict():
    print("=== /predict POST received ===")
    try:
        if 'image' not in request.files:
            return jsonify({'status': 'error', 'message': 'No image provided'}), 400

        image_file = request.files['image']
        print(f"Received image filename: {image_file.filename}")
        print(f"Content-Type: {image_file.content_type}")
        print(f"Content-Length: {request.content_length}")
        image_bytes = image_file.read()

        try:
            image = Image.open(io.BytesIO(image_bytes)).convert('RGB')
        except Exception as e:
            logging.error(f"Failed to open image: {e}")
            return jsonify({'status': 'error', 'message': 'Invalid image format'}), 400

        # Predict articleType with single-task model
        article_type_pred = single_predictor.predict_attributes_from_pil(image)
        article_type_value = article_type_pred.get('articleType')

        # Predict other attributes with multi-task model
        fashion_preds = multi_predictor.predict_attributes_from_pil(image)

        # Remove articleType from multi-task predictions to avoid duplication
        fashion_preds.pop('articleType', None)

        # Combine results: articleType from single-task + other attrs from multi-task
        combined_preds = {**fashion_preds, 'articleType': article_type_value}

        print(f"Prediction results sent: {combined_preds}")

        return jsonify({'status': 'success', 'predictions': combined_preds})

    except Exception as e:
        logging.error(f"Unexpected error: {e}\n{traceback.format_exc()}")
        return jsonify({'status': 'error', 'message': 'Internal server error'}), 500


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, threaded=True)
