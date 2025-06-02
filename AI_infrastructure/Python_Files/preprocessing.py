import tensorflow as tf
from sklearn.utils import shuffle
import numpy as np

class DataPreprocessor:
    def __init__(self, target_size_ef=(300, 300), target_size_iv3=(299, 299)):
        self.target_size_ef = target_size_ef
        self.target_size_iv3 = target_size_iv3
    

    def preprocess_image_ef(self, image):
        # Resize and normalize
        image = image.resize(self.target_size_ef)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        return image_array
    
    def preprocess_image_iv3(self, image):
        # Resize and normalize
        image = image.resize(self.target_size_iv3)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        return image_array
        
    def create_balanced_dataset(self, metadata, target_column, samples_per_class=500):
        balanced_df = (
            metadata.groupby(target_column, group_keys=False)
            .apply(lambda x: x.sample(n=min(len(x), samples_per_class), random_state=42))
            .reset_index(drop=True)
        )
        return balanced_df


    def prepare_ef_dataset(self, image_ids, metadata, loader):
        images = []
        labels = {col: [] for col in ['masterCategory', 'subCategory', 'articleType', 
                                     'baseColour', 'gender', 'season', 'usage']}
        
        for img_id in image_ids:
            img = loader.load_image(img_id)
            if img is None:
                continue
                
            img_data = self.preprocess_image_ef(img)
            images.append(img_data)
            
            row = metadata[metadata['id'] == img_id].iloc[0]
            for col in labels.keys():
                labels[col].append(row[col])
        
        return np.array(images), labels
    
    def prepare_iv3_dataset(self, image_ids, metadata, loader):
        images = []
        labels = {col: [] for col in ['masterCategory', 'subCategory', 'articleType', 
                                     'baseColour', 'gender', 'season', 'usage']}
        
        for img_id in image_ids:
            img = loader.load_image(img_id)
            if img is None:
                continue
                
            img_data = self.preprocess_image_iv3(img)
            images.append(img_data)
            
            row = metadata[metadata['id'] == img_id].iloc[0]
            for col in labels.keys():
                labels[col].append(row[col])
        
        return np.array(images), labels