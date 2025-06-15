import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB0, EfficientNetB1
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.efficientnet import preprocess_input as ef_preprocess
from tensorflow.keras.layers import Resizing


class EfficientNetArtTypeClothingClassifier:
    def __init__(self, num_classes, model_name, input_shape=(300, 300, 3)):
        self.num_classes = num_classes
        self.model_name = model_name
        self.input_shape = input_shape
        if model_name == 'efficientnetB0_artType':
            self.base_model = EfficientNetB0(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        elif model_name == 'efficientnetB1_artType':
            self.base_model = EfficientNetB1(
                weights='imagenet',
                include_top=False,
                input_shape=input_shape
            )
        else:
            self.base_model = EfficientNetB3(
                weights='imagenet', 
                include_top=False, 
                input_shape=input_shape
            )
        
    def build_fine_tuned_model(self, resume=False):
        # Unfreeze all layers except BatchNorm
        for layer in self.base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

        target_size = (224, 224) if self.model_name == 'efficientnetB0_artType' else \
              (240, 240) if self.model_name == 'efficientnetB1_artType' else \
              None

        # Create input
        inputs = tf.keras.Input(shape=self.input_shape)
        x = inputs
        if target_size:
            x = Resizing(*target_size)(inputs)
        x = ef_preprocess(x)
        x = self.base_model(x, training=True)
        x = GlobalAveragePooling2D()(x)
        if self.model_name == 'efficientnetB0_artType':
            x = Dropout(0.2)(x)
        elif self.model_name == 'efficientnetB1_artType':
            x = Dropout(0.3)(x)
        else:
            x = Dropout(0.4)(x)
        if self.model_name == 'efficientnetB0_artType':
            output = Dense(self.num_classes, activation='softmax', name='articleType')(x)
        elif self.model_name == 'efficientnetB1_artType':
            output = Dense(self.num_classes, activation='softmax', name='articleType')(x)
        else:
            output = Dense(self.num_classes, activation='softmax', name='articleType', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)

        # Create model
        model = Model(inputs=inputs, outputs=output)
        if resume:
            # Compile model
            optimizer = SGD(learning_rate=0.00025, momentum=0.9)
        else:
            optimizer = SGD(learning_rate=0.001, momentum=0.9)
        
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Model output:", model.output_names)

        return model