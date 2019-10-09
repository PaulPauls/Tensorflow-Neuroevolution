import numpy as np
import tensorflow as tf
from absl import logging

from ..base_environment import BaseEnvironment


class XOREnvironment(BaseEnvironment):
    """
    Environment for the TFNE framework that represents the simulation of the XOR function and judges genomes based on
    their accuracy in predicting the XOR function. This environment does not train the weights of the supplied genomes
    and their phenotype models in any way.
    """

    def __init__(self):
        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        self.input_shape = (2,)
        self.num_output = 1

    def eval_genome_fitness(self, genome) -> float:
        """
        Calculate and return the genome fitness as the accuracy in its ability to predict the XOR function, rounded to
        3 decimal places.
        :param genome: genome of the TFNE framework, providing a built Tensorflow model
        :return: genome model accuracy of predicting the XOR function, rounded to 3 decimal places
        """
        model = genome.get_model()
        evaluated_fitness = float(100 * (1 - self.loss_function(self.y, model.predict(self.x))))
        return round(evaluated_fitness, 3)

    def replay_genome(self, genome):
        """
        Replay the genome by demonstrating its ability to solve the environment. First show the correct solutions to all
        possible XOR function inputs and then show the genome's predicted solution to those inputs.
        :param genome: genome of the TFNE framework, providing a built Tensorflow model
        """
        model = genome.get_model()
        logging.info("Replaying Genome {}...".format(genome.get_id()))
        logging.info("Solution Values:\n{}".format(self.y))
        logging.info("Predicted Values:\n{}".format(model.predict(self.x)))

    def get_input_shape(self) -> ():
        """
        :return: one-dimensional tuple specifying the number of inputs for a model supplied to this environment
        """
        return self.input_shape

    def get_num_output(self) -> int:
        return self.num_output
