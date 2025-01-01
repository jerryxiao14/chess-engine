import tensorflow as tf
from keras.api.models import Sequential
from keras.api.layers import Activation, Dense, Dropout, Flatten, Conv2D, BatchNormalization, LeakyReLU, Input
from keras.api.optimizers import Adam 

from keras.api.layers import add as add_layer
from keras.api.models import Model 
#from tensorflow.python.keras.engine.keras_tensor import Kerastensor 
from tensorflow.python.keras.engine.keras_tensor import KerasTensor
from tensorflow.python.types.core import ConcreteFunction

import config

class ModelBuilder:
    """
    Builds neural network architecture
    """

    def __init__(self, input_shape, output_shape):
        """
        Neural network f that takes as input the raw board representation and outputs move probabilities p and a value v:
        f(s) = (p,v), where p is a vector of move probabilities and v is the expected value of the position
        """
        self.input_shape = input_shape
        self.output_shape = output_shape 
        self.nr_hidden_layers = 19  # Alphazero used 19, can change maybe later
        self.convolution_filters = config.CONVOLUTION_FILTERS

        
    def build_convolutional_layer(self, input_layer):
        # Add a convolution layer with 256 convolution filters, (3,3) kernel size with stride 1
        layer = Conv2D(filters = self.convolution_filters, kernel_size = (3,3),strides = (1,1), padding = 'same', data_format = 'channels_first',use_bias = False)(input_layer)
        
        # Add batch normalization
        layer = BatchNormalization(axis = 1)(layer)
        # Add Relu activation to the layer 
        layer = Activation('relu')(layer)

        return (layer)
    
    def build_residual_layer(self,input_layer):
        # First build a convolutional layer
        layer = self.build_convolutional_layer(input_layer)

        # Build another convolutional layer with skip connection to erase vanishing gradients
        layer = Conv2D(filters = self.convolution_filters, kernel_size = (3,3), strides = (1,1), padding = 'same', data_format = 'channels_first', use_bias = False)(layer)
        layer = BatchNormalization(axis = 1)(layer)

        #skip connection
        layer = add_layer([layer,input_layer])
        
        #Add relu activation
        layer = Activation('relu')(layer)
        return (layer)
    
    def build_value_head(self) -> Model:
        """
        Builds the value head of the neural network
        """

        model = Sequential(name = 'value_head')
        model.add(Conv2D(1,kernel_size=(1,1),strides = (1,1),
                         input_shape = (self.convolution_filters, self.input_shape[1],self.input_shape[2]),
                         padding = 'same',data_format='channels_first'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Dense(256))
        model.add(Activation('relu'))

        model.add(Dense(self.output_shape[1],
                        activation='tanh',name='value_head'))

        return model
    
    def build_policy_head(self):
        model = Sequential(name='policy_head')
        model.add(Conv2D(1,kernel_size=(1,1),strides = (1,1),
                         input_shape = (self.convolution_filters, self.input_shape[1],self.input_shape[2]),
                         padding = 'same',data_format='channels_first'))
        model.add(BatchNormalization(axis=1))
        model.add(Activation('relu'))
        model.add(Flatten())
        model.add(Activation('relu'))

        model.add(Dense(self.output_shape[0],
                        activation='sigmoid',name='policy_head'))
        return model 
    
    def build_model(self):
        input = Input(shape = self.input_shape, name = 'input')

        x = self.build_convolutional_layer(input)

        # add residual blocks 
        for _ in range(self.nr_hidden_layers):
            x = self.build_residual_layer(x)
        
        model = Model(inputs = input, outputs = x)

        policy_head = self.build_policy_head()
        value_head = self.build_value_head()

        model = Model(inputs = input, outputs = [policy_head(x),value_head(x)])

        model.compile(
            loss = {
                'policy_head':'categorical_crossentropy',
                'value_head': 'mean_squared_error'
            },
            optimizer = Adam(learning_rate=config.LEARNING_RATE),
            loss_weights = {
                'policy_head':0.5,
                'value_head':0.5
            }
        )

        return model
