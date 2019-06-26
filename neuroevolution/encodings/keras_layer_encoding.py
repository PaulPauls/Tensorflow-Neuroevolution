import tensorflow as tf

from neuroevolution.encodings import BaseEncoding
from neuroevolution.encodings import BaseGenome


class KerasLayerModel(tf.keras.Model):

    def __init__(self, input_shape, num_output):
        """
        ToDo
        :param input_shape:
        :param num_output:
        """
        super(KerasLayerModel, self).__init__(name='keras_layer_model')
        self.layer_list = []
        self.layer_list.append(tf.keras.layers.Flatten(input_shape=input_shape))
        self.layer_list.append(tf.keras.layers.Dense(num_output, activation='softmax'))

    def call(self, inputs, **kwargs):
        """
        ToDo
        :param inputs:
        :param kwargs:
        :return:
        """
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x

    def compute_output_signature(self, input_signature):
        """
        ToDo
        :param input_signature:
        :return:
        """
        pass


class KerasLayerGenome(BaseGenome):
    """
    ToDo
    """
    def __init__(self, input_shape, num_output, genome_id):
        """
        ToDo
        :param input_shape:
        :param num_output:
        :param genome_id:
        """
        self.id = genome_id
        self.fitness = 0
        self.model = KerasLayerModel(input_shape, num_output)
        self.compile_model()

    def compile_model(self):
        """
        ToDo
        :return:
        """
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])

    def to_phenotype(self):
        """
        ToDo
        :return:
        """
        return self.model

    def add_layer(self, index, layer):
        """
        ToDo
        :param index:
        :param layer:
        :return:
        """
        self.model.layer_list.insert(index, layer)
        self.compile_model()

    def replace_layer(self, index, layer_type, units=None, activation=None):
        """
        ToDo
        :param index:
        :param layer_type:
        :param units:
        :param activation:
        :return:
        """
        if units is None:
            units = self.model.layer_list[index].units
        if activation is None:
            activation = self.model.layer_list[index].activation
        self.model.layer_list[index] = layer_type(units=units, activation=activation)
        self.compile_model()

    def get_layer_count(self):
        """
        ToDo
        :return:
        """
        return len(self.model.layer_list)

    def set_id(self, genome_id):
        """
        ToDo
        :param genome_id:
        :return:
        """
        self.id = genome_id

    def get_id(self):
        """
        ToDo
        :return:
        """
        return self.id

    def set_fitness(self, fitness):
        """
        ToDo
        :param fitness:
        :return:
        """
        self.fitness = fitness

    def get_fitness(self):
        """
        ToDo
        :return:
        """
        return self.fitness


class KerasLayerEncoding(BaseEncoding):
    """
    ToDo
    """
    def __init__(self, input_shape, num_output, config):
        """
        ToDo
        :param input_shape:
        :param num_output:
        :param config:
        """
        self.input_shape = input_shape
        self.num_output = num_output

        self.genome_id_counter = 0

    def create_genome(self):
        """
        ToDo
        :return:
        """
        genome = KerasLayerGenome(self.input_shape, self.num_output, self.genome_id_counter)
        self.genome_id_counter += 1
        return genome

    def pop_id(self):
        """
        ToDo
        :return:
        """
        new_id = self.genome_id_counter
        self.genome_id_counter += 1
        return new_id
