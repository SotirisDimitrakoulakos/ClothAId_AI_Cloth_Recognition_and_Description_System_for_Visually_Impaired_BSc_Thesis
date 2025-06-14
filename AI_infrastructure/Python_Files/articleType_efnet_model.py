import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.applications.efficientnet import preprocess_input as ef_preprocess


class EfficientNetArtTypeClothingClassifier:
    def __init__(self, num_classes, input_shape=(300, 300, 3)):
        self.num_classes = num_classes
        self.input_shape = input_shape
        self.base_model = EfficientNetB3(
            weights='imagenet', 
            include_top=False, 
            input_shape=input_shape
        )
        
    def build_fine_tuned_model(self):

        # Unfreeze all layers except BatchNorm
        for layer in self.base_model.layers:
            if isinstance(layer, tf.keras.layers.BatchNormalization):
                layer.trainable = False
            else:
                layer.trainable = True

        # Create input
        inputs = tf.keras.Input(shape=self.input_shape)
        x = ef_preprocess(inputs)
        x = self.base_model(x, training=True)
        x = GlobalAveragePooling2D()(x)
        x = Dropout(0.4)(x)
        

        output = Dense(self.num_classes, activation='softmax', name='articleType', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
        
        # Create model
        model = Model(inputs=inputs, outputs=output)
        
        # Compile model
        optimizer = SGD(learning_rate=0.001, momentum=0.9)
        
        model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
        print("Model output:", model.output_names)

        return model