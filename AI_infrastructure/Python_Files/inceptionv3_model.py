import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as iv3_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD
from focal_loss import SparseFocalLoss

class IV3ClothingClassifier:
    def __init__(self, num_classes_dict, input_shape=(299, 299, 3)):
        self.num_classes_dict = num_classes_dict
        self.input_shape = input_shape
        self.base_model = InceptionV3(  # Load pre-trained InceptionV3
            weights='imagenet',          # Pre-trained on ImageNet
            include_top=False,           # Remove original classification head
            input_shape=self.input_shape
        )
        
    def build_model(self):
        # Freeze base model layers
        self.base_model.trainable = False
        
        # Explicitly create input layer
        inputs = tf.keras.Input(shape=self.input_shape)
        x = iv3_preprocess(inputs)
        x = self.base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        
        # Create multi-output branches
        outputs = {}
        for attr, num_classes in self.num_classes_dict.items():
            outputs[attr] = Dense(num_classes, activation='softmax', name=attr)(x)
            
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile
        optimizer = SGD(learning_rate=0.001, momentum=0.9)
        losses = {
            'season': SparseFocalLoss(gamma=2.0),
            'baseColour': SparseFocalLoss(gamma=2.0),
            'articleType': 'sparse_categorical_crossentropy',
            'subCategory': 'sparse_categorical_crossentropy',
            'masterCategory': 'sparse_categorical_crossentropy',
            'gender': 'sparse_categorical_crossentropy',
            'usage': 'sparse_categorical_crossentropy',
        }
        loss_weights = {
            'articleType': 1.2,
            'subCategory': 1.0,
            'baseColour': 1.5,
            'season': 1.5,
            'usage': 1.0,
            'masterCategory': 0.8,
            'gender': 0.8
        }
        metrics = {attr: 'accuracy' for attr in self.num_classes_dict.keys()}
        
        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
        return model
    
    def fine_tune(self, model, unfreeze_from_layer_name='mixed7'):
        # Unfreeze from specific layer onward in base model
        unfreeze = False
        for layer in self.base_model.layers:
            if layer.name == unfreeze_from_layer_name:
                unfreeze = True
            if unfreeze:
                if not isinstance(layer, tf.keras.layers.BatchNormalization):
                    layer.trainable = True
                else:
                    layer.trainable = False
            else:
                layer.trainable = False

        # Ensure all layers outside the base model (i.e., heads) are trainable
        for layer in model.layers:
            if layer not in self.base_model.layers:
                layer.trainable = True

        model.trainable = True  # Important safety net
            
        # Lower learning rate for fine-tuning
        optimizer = SGD(learning_rate=0.0001, momentum=0.9)
        losses = {
            'season': SparseFocalLoss(gamma=2.0),
            'baseColour': SparseFocalLoss(gamma=2.0),
            'articleType': 'sparse_categorical_crossentropy',
            'subCategory': 'sparse_categorical_crossentropy',
            'masterCategory': 'sparse_categorical_crossentropy',
            'gender': 'sparse_categorical_crossentropy',
            'usage': 'sparse_categorical_crossentropy',
        }
        loss_weights = {
            'articleType': 1.2,
            'subCategory': 1.0,
            'baseColour': 1.5,
            'season': 1.5,
            'usage': 1.0,
            'masterCategory': 0.8,
            'gender': 0.8
        }
        metrics = {attr: 'accuracy' for attr in self.num_classes_dict.keys()}
        
        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
        return model
