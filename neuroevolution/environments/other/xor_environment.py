import ast
import numpy as np
import tensorflow as tf
from absl import logging

from neuroevolution.environments import BaseEnvironment


class XOREnvironment(BaseEnvironment):
    """
    ToDo: doc
    """

    def __init__(self, config):
        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])
        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        # Read in config parameters for the xor environment
        section_name = 'XOR_ENVIRONMENT' if config.has_section('XOR_ENVIRONMENT') else 'ENVIRONMENT'
        self.input_shape = ast.literal_eval(config.get(section_name, 'input_shape', fallback='(2,)'))
        self.num_output = config.getint(section_name, 'num_output', fallback=1)

        logging.debug("XOR Environment read from config: input_shape = {}".format(self.input_shape))
        logging.debug("XOR Environment read from config: num_output = {}".format(self.num_output))

    def eval_genome_fitness(self, genome):
        """
        ToDo: doc
        """
        # Get the phenotype model from the genome
        model = genome.get_phenotype_model()

        # Calculate the genome fitness as the percentage of accuracy in its prediction, rounded to 3 decimal points
        evaluated_fitness = float(100 * (1 - self.loss_function(self.y, model.predict(self.x))))
        rounded_evaluated_fitness = round(evaluated_fitness, 3)
        genome.set_fitness(rounded_evaluated_fitness)

    def replay_genome(self, genome):
        model = genome.get_phenotype_model()
        print("#" * 100)
        print("Solution Values:\n{}".format(self.y))
        print("Predicted Values:\n{}".format(model.predict(self.x)))
        print("#" * 100)

    def get_input_shape(self):
        return self.input_shape

    def get_num_output(self):
        return self.num_output


class XORWeightTrainingEnvironment(BaseEnvironment):
    """
    ToDo: doc
    """

    def __init__(self, config):
        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])
        self.loss_function = tf.keras.losses.BinaryCrossentropy()

        # Read in config parameters for the xor environment
        section_name = 'XOR_ENVIRONMENT' if config.has_section('XOR_ENVIRONMENT') else 'ENVIRONMENT'
        self.input_shape = ast.literal_eval(config.get(section_name, 'input_shape', fallback='(2,)'))
        self.num_output = config.getint(section_name, 'num_output', fallback=1)
        self.learning_rate = config.getfloat(section_name, 'learning_rate')
        self.epochs = config.getint(section_name, 'epochs')
        self.early_stop = config.getboolean(section_name, 'early_stop')
        if self.early_stop:
            self.early_stop_min_delta = config.getfloat(section_name, 'early_stop_min_delta')
            self.early_stop_patience = config.getint(section_name, 'early_stop_patience')

        logging.debug("Environment read from config: input_shape = {}".format(self.input_shape))
        logging.debug("Environment read from config: num_output = {}".format(self.num_output))
        logging.debug("Environment read from config: learning_rate = {}".format(self.learning_rate))
        logging.debug("Environment read from config: epochs = {}".format(self.epochs))
        logging.debug("Environment read from config: early_stop = {}".format(self.early_stop))
        if self.early_stop:
            logging.debug("Environment read from config: early_stop_min_delta = {}"
                          .format(self.early_stop_min_delta))
            logging.debug("Environment read from config: early_stop_patience = {}".format(self.early_stop_patience))

    def eval_genome_fitness(self, genome):
        """
        ToDo: doc
        """
        # Get the phenotype model from the genome and declare the optimizer
        model = genome.get_phenotype_model()
        optimizer = tf.keras.optimizers.SGD(lr=self.learning_rate)

        # Compile and train the model
        model.compile(optimizer=optimizer, loss=self.loss_function)
        if self.early_stop:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=self.early_stop_min_delta,
                                                          patience=self.early_stop_patience)
            model.fit(self.x, self.y, batch_size=1, epochs=self.epochs, verbose=0, callbacks=[early_stop])
        else:
            model.fit(self.x, self.y, batch_size=1, epochs=self.epochs, verbose=0)

        # Calculate the genome fitness as the percentage of accuracy in its prediction, rounded to 3 decimal points
        evaluated_fitness = float(100 * (1 - self.loss_function(self.y, model.predict(self.x))))
        rounded_evaluated_fitness = round(evaluated_fitness, 3)
        genome.set_fitness(rounded_evaluated_fitness)

    def replay_genome(self, genome):
        model = genome.get_phenotype_model()
        print("#" * 100)
        print("Solution Values:\n{}".format(self.y))
        print("Predicted Values:\n{}".format(model.predict(self.x)))
        print("#" * 100)

    def get_input_shape(self):
        return self.input_shape

    def get_num_output(self):
        return self.num_output
