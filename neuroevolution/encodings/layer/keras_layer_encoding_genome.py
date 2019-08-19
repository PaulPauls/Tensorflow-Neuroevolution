raise NotImplementedError
'''
Not yet compatible with Architecture Refactoring (TFNE Frameowrk still in early Alpha!)
'''

from neuroevolution.encodings import BaseGenome
from neuroevolution.encodings.layer import KerasLayerEncodingModel


class KerasLayerEncodingGenome(BaseGenome):
    def __init__(self, input_shape, num_output, genome_id):
        self.id = genome_id
        self.fitness = 0
        self.input_shape = input_shape
        self.num_output = num_output
        self.model = KerasLayerEncodingModel(input_shape, num_output)
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
