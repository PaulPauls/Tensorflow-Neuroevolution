from typing import Union

import tensorflow as tf


class OptimizerFactory:
    """"""

    def __init__(self, optimizer_parameters):
        """"""
        # Register parameters for optimizer
        self.optimizer_parameters = optimizer_parameters

    def __str__(self) -> str:
        """"""
        return "Optimizer: {} (Config: {})".format(self.optimizer_parameters['class_name'],
                                                   self.optimizer_parameters['config'])

    def create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """"""
        return tf.keras.optimizers.deserialize(self.optimizer_parameters)

    def duplicate(self):
        """"""
        return OptimizerFactory(self.optimizer_parameters)

    def get_parameters(self) -> {str: Union[str, dict]}:
        """"""
        return self.optimizer_parameters

    def get_name(self) -> str:
        """"""
        return self.optimizer_parameters['class_name']
