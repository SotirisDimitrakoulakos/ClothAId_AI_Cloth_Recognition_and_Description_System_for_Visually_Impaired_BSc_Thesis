import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json
import os
import numpy as np
from sklearn.preprocessing import LabelEncoder

class MultiOutputDataGenerator(tf.keras.utils.Sequence):
    def __init__(self, X, y_dict, batch_size=32, shuffle=True):
        self.X = X
        self.y_dict = y_dict
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.indices = np.arange(len(X))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.X) / self.batch_size))
    
    def __getitem__(self, index):
        batch_indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        batch_X = self.X[batch_indices]
        batch_y = {key: val[batch_indices] for key, val in self.y_dict.items()}
        print(f"Batch {index}: X shape {batch_X.shape}, y keys: {list(batch_y.keys())}, y[0] shape: {batch_y[next(iter(batch_y))].shape}")
        return batch_X, batch_y
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

class ClothingClassifierTrainer:
    def __init__(self, model, model_name):
        self.model = model
        self.model_name = model_name
        self.label_encoders = {}
        self.augmentor = ImageDataGenerator(
            horizontal_flip=True,
        )
    
    def fit_label_encoders(self, *label_dicts):
        all_attrs = label_dicts[0].keys()
        for attr in all_attrs:
            combined = []
            for d in label_dicts:
                if d and attr in d:
                    combined += list(d[attr])
            self.label_encoders[attr] = LabelEncoder()
            self.label_encoders[attr].fit(combined)

        
    def encode_labels(self, labels_dict):
        """Transform labels using already fitted encoders."""
        encoded_labels = {}
        for attr, labels in labels_dict.items():
            if attr not in self.label_encoders:
                raise ValueError(f"Label encoder for '{attr}' not fitted yet.")
            encoded_labels[attr] = np.array(self.label_encoders[attr].transform(labels))
        return encoded_labels
    
    def train(self, X_train, y_train, X_val, y_val, batch_size=16, epochs=50, X_test=None, y_test=None):

        # Fit label encoders on all available labels before encoding
        self.fit_label_encoders(y_train, y_val, y_test)

       # Encode labels
        y_train_encoded = self.encode_labels(y_train)
        y_val_encoded = self.encode_labels(y_val)

        print("Encoded y_train keys and shapes:")
        for k, v in y_train_encoded.items():
            print(f"{k}: {v.shape}")

        print("Encoded y_val keys and shapes:")
        for k, v in y_val_encoded.items():
            print(f"{k}: {v.shape}")

        # Convert encoded labels dict into a NumPy array for multi-output (if needed)
        # For training generator
        y_train_array = {attr: y_train_encoded[attr] for attr in y_train_encoded}

        # For validation generator
        y_val_array = {attr: y_val_encoded[attr] for attr in y_val_encoded}

        # Create data generators
        train_gen = MultiOutputDataGenerator(X_train, y_train_array, batch_size=batch_size)
        val_gen = MultiOutputDataGenerator(X_val, y_val_array, batch_size=batch_size, shuffle=False)

        
        # Callbacks
        callbacks = [
            ModelCheckpoint(
                f'{self.model_name}_best_weights.h5',
                monitor='val_loss',
                save_best_only=True,
                save_weights_only=True
            ),
            EarlyStopping(
                monitor='val_loss',
                patience=5,
                restore_best_weights=True
            )
        ]
        
        # Train model
        history = self.model.fit(
            train_gen,
            validation_data=val_gen,
            batch_size=batch_size,
            epochs=epochs,
            callbacks=callbacks
        )
        
        return history
    
    def save_model(self, save_dir):
        # Save model weights
        self.model.save_weights(os.path.join(save_dir, f'{self.model_name}_final_weights.h5'))
        
        # Save label encoders
        for attr, encoder in self.label_encoders.items():
            np.save(os.path.join(save_dir, f'{attr}_classes.npy'), encoder.classes_)
        
        # Save model architecture if needed
        with open(os.path.join(save_dir, f'{self.model_name}_architecture.json'), 'w') as f:
            f.write(self.model.to_json())
    
    @staticmethod
    def load_model(model_name, model_arch, save_dir):
        # Load model architecture
        model = model_arch
        
        # Load weights
        model.load_weights(os.path.join(save_dir, f'{model_name}_final_weights.h5'))
        
        # Load label encoders
        label_encoders = {}
        for attr in ['masterCategory', 'subCategory', 'articleType', 
                    'baseColour', 'gender', 'season', 'usage']:
            classes = np.load(os.path.join(save_dir, f'{attr}_classes.npy'))
            encoder = LabelEncoder()
            encoder.classes_ = classes
            label_encoders[attr] = encoder
        
        return model, label_encoders