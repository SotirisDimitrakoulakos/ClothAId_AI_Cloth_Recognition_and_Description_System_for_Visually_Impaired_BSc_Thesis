import tensorflow as tf
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input as iv3_preprocess
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD

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
        losses = {attr: 'sparse_categorical_crossentropy' for attr in self.num_classes_dict.keys()}
        metrics = {attr: 'accuracy' for attr in self.num_classes_dict.keys()}
        
        model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        return model
    
    def fine_tune(self, model, unfreeze_from_layer_name='mixed7'):
        # Freeze all layers before mixed7 (layer 228)
        unfreeze = False
        for layer in model.layers:
            if layer.name == unfreeze_from_layer_name:
                unfreeze = True
            layer.trainable = unfreeze
            
        # Lower learning rate for fine-tuning
        optimizer = SGD(learning_rate=0.0001, momentum=0.9)
        losses = {attr: 'sparse_categorical_crossentropy' for attr in self.num_classes_dict.keys()}
        metrics = {attr: 'accuracy' for attr in self.num_classes_dict.keys()}
        
        model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        return model
