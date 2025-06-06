import numpy as np
from PIL import Image
import tensorflow as tf
from sklearn.preprocessing import LabelEncoder

class ClothingAttributePredictor:
    def __init__(self, model, label_encoders):
        self.model = model
        self.label_encoders = label_encoders
    
    def preprocess_image_from_pil(self, img, target_size=(300, 300)):
        img = img.resize(target_size)
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        img_array = img_array / 255.0  # Normalize
        img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
        return img_array
    
    def predict_attributes_from_pil(self, img):
        img_array = self.preprocess_image_from_pil(img)
        predictions = self.model.predict(img_array)

        # If your model output is a dict (multi-output), else adjust accordingly
        result = {}
        for attr, pred_array in predictions.items():
            pred_class = np.argmax(pred_array[0])
            class_name = self.label_encoders[attr].classes_[pred_class]
            result[attr] = class_name
        
        return result