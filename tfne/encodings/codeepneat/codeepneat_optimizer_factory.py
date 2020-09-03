from typing import Union

import tensorflow as tf


class OptimizerFactory:
    """
    Optimizer factory of the CoDeepNEAT encoding, serving to easily create identical though seperate TF optimizer
    instances.
    """

    def __init__(self, optimizer_parameters):
        """
        Create the optimizer factory by saving the parameters with which the created TF optimizers will be initialized
        @param optimizer_parameters: dict that can be deserialized by TF to a valid TF optimizer
        """
        # Register parameters for optimizer
        self.optimizer_parameters = optimizer_parameters

    def __str__(self) -> str:
        """
        @return: string representation of the optimizer
        """
        return "Optimizer: {} (Config: {})".format(self.optimizer_parameters['class_name'],
                                                   self.optimizer_parameters['config'])

    def create_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """
        @return: newly created deserialized instance of a TF optimizer
        """
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
