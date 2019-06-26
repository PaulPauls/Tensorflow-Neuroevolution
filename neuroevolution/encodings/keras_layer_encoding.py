import ast
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
        """
        self.id = genome_id
        self.fitness = 0
        self.model = KerasLayerModel(input_shape, num_output)

    def to_phenotype(self):
        """
        ToDo
        :return:
        """
        self.model.compile(optimizer='adam',
                           loss='sparse_categorical_crossentropy',
                           metrics=['accuracy'])
        return self.model

    def get_id(self):
        """
        ToDo
        :return:
        """
        return self.id

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
        """
        self.input_shape = input_shape
        self.num_output = num_output

        # Read in config parameters for genome encoding
        self.available_activations = ast.literal_eval(config.get('KerasLayerEncoding', 'available_activations'))

    def create_genome(self, genome_id):
        """
        ToDo
        :param: genome_id
        :return:
        """
        return KerasLayerGenome(self.input_shape, self.num_output, genome_id)
