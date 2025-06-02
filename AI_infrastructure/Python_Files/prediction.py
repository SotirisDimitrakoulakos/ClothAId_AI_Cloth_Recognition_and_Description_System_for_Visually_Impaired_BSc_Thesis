import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

class ClothingAttributePredictor:
    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders
    
    def preprocess_image(self, image_path, target_size=(300, 300)):
        img = Image.open(image_path)
        img = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    
    def predict_attributes(self, image_path):
        # Preprocess image
        img_array = self.preprocess_image(image_path)
        
        # Predict
        predictions = self.model.predict(img_array)
        
        # Decode predictions
        result = {}
        for attr in predictions.keys():
            pred_class = np.argmax(predictions[attr][0])
            class_name = self.label_encoders[attr].classes_[pred_class]
            result[attr] = class_name
        
        return result