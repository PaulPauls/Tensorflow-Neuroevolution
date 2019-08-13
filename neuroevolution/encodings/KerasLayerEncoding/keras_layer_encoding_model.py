raise NotImplementedError
'''
Not yet compatible with Architecture Refactoring (TFNE Frameowrk still in early Alpha!)
'''

import tensorflow as tf


class KerasLayerEncodingModel(tf.keras.Model):

    def __init__(self, input_shape, num_output):
        super(KerasLayerEncodingModel, self).__init__(name='keras_layer_model')
        self.layer_list = []
        self.layer_list.append(tf.keras.layers.Flatten(input_shape=input_shape))
        self.layer_list.append(tf.keras.layers.Dense(num_output, activation='softmax'))

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x

    def compute_output_signature(self, input_signature):
        # ToDo
        pass
