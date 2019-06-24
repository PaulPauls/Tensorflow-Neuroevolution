import tensorflow as tf

from neuroevolution.encodings.base_genome import BaseGenome


class KerasLayerEncodingModel(tf.keras.Model):

    def __init__(self, input_shape, num_output):
        super(KerasLayerEncodingModel, self).__init__(name='keras_layer_encoding_model')
        self.layer_list = []
        self.layer_list.append(tf.keras.layers.Flatten(input_shape=input_shape))
        self.layer_list.append(tf.keras.layers.Dense(num_output, activation='softmax'))

    def call(self, inputs, **kwargs):
        x = inputs
        for layer in self.layer_list:
            x = layer(x)
        return x


class KerasLayerEncodingGenome(BaseGenome):
    """
    ToDo
    """
    def __init__(self, input_shape, num_output):
        """
        ToDo
        """
        self.fitness = 0
        self.id = None
        self.phenotype = KerasLayerEncodingModel(input_shape, num_output)
        self.genotype = self.phenotype

        self.phenotype.compile(optimizer='adam',
                      loss='sparse_categorical_crossentropy',
                      metrics=['accuracy'])

    def translate_to_phenotype(self):
        """
        ToDo
        :return:
        """
        return self.phenotype

    def set_id(self, id):
        """
        ToDo
        :param: id
        :return:
        """
        self.id = id
