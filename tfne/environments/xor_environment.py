from __future__ import annotations

import numpy as np
import tensorflow as tf

from .base_environment import BaseEnvironment
from tfne.helper_functions import read_option_from_config


class XOREnvironment(BaseEnvironment):
    """
    TFNE compatible environment for the XOR problem
    """

    def __init__(self, weight_training, config=None, verbosity=0, **kwargs):
        """
        Initializes XOR environment by setting up the dataset and processing the supplied config or supplied config
        parameters. The configuration of the environment can either be supplied via a config file or via seperate config
        parameters in the initialization.
        @param weight_training: bool flag, indicating wether evaluation should be weight training or not
        @param config: ConfigParser instance holding an 'Environment' section specifying the required environment
                       parameters for the chosen evaluation method.
        @param verbosity: integer specifying the verbosity of the evaluation
        @param kwargs: Optionally supplied dict of each configuration parameter seperately in order to allow the
                       creation of the evaluation environment without the requirement of a config file.
        """
        # Initialize corresponding input and output mappings
        print("Setting up XOR environment...")
        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

        # Initialize loss function to evaluate performance on either evaluation method and safe verbosity parameter
        self.loss_function = tf.keras.losses.BinaryCrossentropy()
        self.verbosity = verbosity

        # Determine and setup explicit evaluation method in accordance to supplied parameters
        if not weight_training:
            # Set up XOR environment as non-weight training, requiring no parameters
            self.eval_genome_fitness = self._eval_genome_fitness_non_weight_training

        elif config is None and len(kwargs) == 0:
            raise RuntimeError("XOR environment is being set up as weight training, though neither config file nor "
                               "explicit config parameters for the weight training were supplied")

        elif len(kwargs) == 0:
            # Set up XOR environment as weight training and with a supplied config file
            self.eval_genome_fitness = self._eval_genome_fitness_weight_training
            self.epochs = read_option_from_config(config, 'EVALUATION', 'epochs')
            self.batch_size = read_option_from_config(config, 'EVALUATION', 'batch_size')

        elif config is None:
            # Set up XOR environment as weight training and explicitely supplied parameters
            self.eval_genome_fitness = self._eval_genome_fitness_weight_training
            self.epochs = kwargs['epochs']
            self.batch_size = kwargs['batch_size']

    def eval_genome_fitness(self, genome) -> float:
        # TO BE OVERRIDEN
        raise RuntimeError()

    def _eval_genome_fitness_weight_training(self, genome) -> float:
        """
        Evaluates the genome's fitness by obtaining the associated Tensorflow model and optimizer, compiling them and
        then training them for the config specified duration. The genomes fitness is then calculated and returned as
        the binary cross entropy in percent of the predicted to the actual results
        @param genome: TFNE compatible genome that is to be evaluated
        @return: genome calculated fitness
        """
        # Get model and optimizer required for compilation
        model = genome.get_model()
        optimizer = genome.get_optimizer()

        # Compile and train model
        model.compile(optimizer=optimizer, loss=self.loss_function)
        model.fit(x=self.x, y=self.y, epochs=self.epochs, batch_size=self.batch_size, verbose=self.verbosity)

        # Evaluate and return its fitness
        evaluated_fitness = float(100 * (1 - self.loss_function(self.y, model(self.x))))

        # FIXME Tensorflow arbitrary NaN loss when using float16 datatype. Confirmed by TF.
        # Github TF issue: https://github.com/tensorflow/tensorflow/issues/38457
        if tf.math.is_nan(evaluated_fitness):
            evaluated_fitness = 0

        return round(evaluated_fitness, 4)

    def _eval_genome_fitness_non_weight_training(self, genome) -> float:
        """
        Evaluates genome's fitness by calculating and returning the binary cross entropy in percent of the predicted to
        the actual results
        @param genome: TFNE compatible genome that is to be evaluated
        @return: genome calculated fitness
        """
        # Evaluate and return its fitness by calling genome directly with input
        evaluated_fitness = float(100 * (1 - self.loss_function(self.y, genome(self.x))))
        return round(evaluated_fitness, 4)

    def replay_genome(self, genome):
        """
        Replay genome on environment by calculating its fitness and printing it.
        @param genome: TFNE compatible genome that is to be evaluated
        """
        print("Replaying Genome #{}:".format(genome.get_id()))
        evaluated_fitness = round(float(100 * (1 - self.loss_function(self.y, genome(self.x)))), 4)
        print("Solution Values: \t{}\n".format(self.y))
        print("Predicted Values:\t{}\n".format(genome(self.x)))
        print("Achieved Fitness:\t{}\n".format(evaluated_fitness))

    def duplicate(self) -> XOREnvironment:
        """
        @return: New instance of the XOR environment with identical parameters
        """
        if hasattr(self, 'epochs'):
            return XOREnvironment(True, verbosity=self.verbosity, epochs=self.epochs, batch_size=self.batch_size)
        else:
            return XOREnvironment(False, verbosity=self.verbosity)

    def get_input_shape(self) -> (int,):
        """"""
        return (2,)

    def get_output_shape(self) -> (int,):
        """"""
        return (1,)
