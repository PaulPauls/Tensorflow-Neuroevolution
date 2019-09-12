import tensorflow as tf
from toposort import toposort


class DirectEncodingModel(tf.keras.Model):
    def __init__(self, genotype, trainable, dtype, run_eagerly):
        super(DirectEncodingModel, self).__init__(trainable=trainable, dtype=dtype)
        self.run_eagerly = run_eagerly

        raise NotImplementedError()

    def call(self, inputs):
        raise NotImplementedError()
