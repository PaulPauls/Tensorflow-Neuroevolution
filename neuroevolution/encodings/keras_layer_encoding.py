raise NotImplementedError
'''
Not yet compatible with Architecture Refactoring (TFNE still in Alpha)
'''

import tensorflow as tf

from neuroevolution.encodings import BaseEncoding
from neuroevolution.encodings import BaseGenome


class KerasLayerModel(tf.keras.Model):

    def __init__(self, input_shape, num_output):
        super(KerasLayerModel, self).__init__(name='keras_layer_model')
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


class KerasLayerGenome(BaseGenome):
    def __init__(self, input_shape, num_output, genome_id):
        self.id = genome_id
        self.fitness = 0
        self.input_shape = input_shape
        self.num_output = num_output
        self.model = KerasLayerModel(input_shape, num_output)
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def to_phenotype(self):
        return self.model

    def add_layer(self, index, layer):
        self.model.layer_list.insert(index, layer)
        self._build_layer(index)

    def replace_layer(self, index, layer_type, units=None, activation=None):
        if units is None:
            units = self.model.layer_list[index].units
        if activation is None:
            activation = self.model.layer_list[index].activation
        self.model.layer_list[index] = layer_type(units=units, activation=activation)
        self._build_layer(index)

    def _build_layer(self, index):
        # Get output shape of preceding layer
        input_shape = self.input_shape
        for layer_index in range(index-1):
            input_shape = self.model.layer_list[layer_index].compute_output_shape(input_shape)

        self.model.layer_list[index].build(input_shape)

    def get_layer_count(self):
        return len(self.model.layer_list)

    def set_id(self, genome_id):
        self.id = genome_id

    def get_id(self):
        return self.id

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness


class KerasLayerEncoding(BaseEncoding):
    def __init__(self, input_shape, num_output, config):
        self.input_shape = input_shape
        self.num_output = num_output

        self.genome_id_counter = 0

    def create_genome(self):
        genome = KerasLayerGenome(self.input_shape, self.num_output, self.genome_id_counter)
        self.genome_id_counter += 1
        return genome

    def pop_id(self):
        new_id = self.genome_id_counter
        self.genome_id_counter += 1
        return new_id
