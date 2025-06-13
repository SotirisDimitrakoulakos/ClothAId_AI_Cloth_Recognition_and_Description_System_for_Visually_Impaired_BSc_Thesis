import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.efficientnet import preprocess_input as ef_preprocess


class EfficientNetClothingClassifier:
    def __init__(self, num_classes_dict, input_shape=(300, 300, 3)):
        self.num_classes_dict = num_classes_dict
        self.input_shape = input_shape
        self.base_model = EfficientNetB3(
            weights='imagenet', 
            include_top=False, 
            input_shape=input_shape
        )
        self.models = {}
        
    def build_model(self):
        # Freeze base model layers
        self.base_model.trainable = False
        
        # Create input
        inputs = tf.keras.Input(shape=self.input_shape)
        x = ef_preprocess(inputs)
        x = self.base_model(x, training=False)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.2)(x)
        
        # Create output branches for each attribute
        outputs = {}
        for attr, num_classes in self.num_classes_dict.items():
            outputs[attr] = Dense(num_classes, activation='softmax', name=attr)(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=outputs)
        
        # Compile model
        optimizer = SGD(learning_rate=0.001, momentum=0.9)
        losses = {attr: 'sparse_categorical_crossentropy' for attr in self.num_classes_dict}
        metrics = {attr: 'accuracy' for attr in self.num_classes_dict.keys()}

        print("Model outputs:", model.output_names)
        print("Loss keys:", list(losses.keys()))
        
        model.compile(optimizer=optimizer, loss=losses, metrics=metrics)
        
        return model
    
    def fine_tune(self, model, unfreeze_from_layer_name='block5e_expand_conv'):
        # Unfreeze from specified layer, keep BatchNorm frozen
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
            
        # Recompile model
        optimizer = SGD(learning_rate=0.0001, momentum=0.9)  # Lower learning rate for fine-tuning
        losses = {attr: 'sparse_categorical_crossentropy' for attr in self.num_classes_dict}
        loss_weights = {
            'articleType': 1.2,
            'subCategory': 0.8,
            'baseColour': 1.5,
            'season': 1.5,
            'usage': 1.0,
            'masterCategory': 0.7,
            'gender': 1.0
        }
        metrics = {attr: 'accuracy' for attr in self.num_classes_dict.keys()}
        
        model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics)
        
        return model