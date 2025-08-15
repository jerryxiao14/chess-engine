import tensorflow as tf
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, BatchNormalization, Activation, Flatten, Dense, Add
)
import config
import os

class ModelBuilder:
    def __init__(self, input_shape, output_shape):
        self.input_shape = input_shape  # e.g., (8, 8, 19) NHWC
        self.output_shape = output_shape  # e.g., (num_moves, 1)
        self.nr_hidden_layers = 19
        self.convolution_filters = config.CONVOLUTION_FILTERS

    def build_convolutional_layer(self, input_layer):
        layer = Conv2D(
            filters=self.convolution_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False
        )(input_layer)
        layer = BatchNormalization(axis=-1)(layer)
        layer = Activation('relu')(layer)
        return layer

    def build_residual_layer(self, input_layer):
        layer = self.build_convolutional_layer(input_layer)
        layer = Conv2D(
            filters=self.convolution_filters,
            kernel_size=(3, 3),
            strides=(1, 1),
            padding='same',
            use_bias=False
        )(layer)
        layer = BatchNormalization(axis=-1)(layer)
        layer = Add()([layer, input_layer])
        layer = Activation('relu')(layer)
        return layer

    def build_value_head(self):
        model = Sequential(name='value_head')
        model.add(Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(256, activation='relu'))
        model.add(Dense(self.output_shape[1], activation='tanh', name='value_head'))
        return model

    def build_policy_head(self):
        model = Sequential(name='policy_head')
        model.add(Conv2D(1, kernel_size=(1, 1), strides=(1, 1), padding='same'))
        model.add(BatchNormalization(axis=-1))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(self.output_shape[0], activation='sigmoid', name='policy_head'))
        return model

    def build_model(self):
        input_layer = Input(shape=self.input_shape, name='input')

        x = self.build_convolutional_layer(input_layer)
        for _ in range(self.nr_hidden_layers):
            x = self.build_residual_layer(x)

        policy_head = self.build_policy_head()
        value_head = self.build_value_head()

        # Connect heads to the trunk output
        outputs = [policy_head(x), value_head(x)]

        model = Model(inputs=input_layer, outputs=outputs)
        model.compile(
            loss={
                'policy_head': 'categorical_crossentropy',
                'value_head': 'mean_squared_error'
            },
            optimizer=tf.keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
            loss_weights={
                'policy_head': 0.5,
                'value_head': 0.5
            }
        )
        return model


if __name__ == "__main__":
    model_builder = ModelBuilder(input_shape=config.INPUT_SHAPE, output_shape=config.OUTPUT_SHAPE)
    model = model_builder.build_model()

    if not os.path.exists("models"):
        os.makedirs("models")

    model.save(os.path.join("models", "base_model") + '.keras')
    print("Model successfully built and saved.")
