import ast
import numpy as np
import tensorflow as tf

from neuroevolution.environments import BaseEnvironment


class XOREnvironment(BaseEnvironment):
    """
    ToDo: Redo docstring
    Environment that provides the XOR function imitation problem to the TFNE framework in a way that assumes that only
    the genome model's topology will be evaluated. In order to reasonably determine the genomes fitness based on the
    topology will this environment first train the weights of the model according to the neuroevolution configuration
    and then assign a fitness score to the genome.
    """

    def __init__(self, config):
        pass

    def eval_genome_fitness(self, genome):
        """
        ToDo: Redo docstring
        Evaluate the genome by grabbing its model and training its weights with SGD. If activated will the training
        process stop early if no progress will take place. The genomes fitness is then determined as the inverted loss
        of the model, multiplied by 100 to get the percentage, effectively representing the model's accuracy in
        predicting the correct values.
        """
        raise NotImplementedError("Should implement eval_genome_fitness()")

    def replay_genome(self, genome):
        raise NotImplementedError("Should implement replay_genome()")

    def get_input_shape(self):
        raise NotImplementedError("Should implement get_input_shape()")

    def get_num_output(self):
        raise NotImplementedError("Should implement get_num_output()")


class XORWeightTrainingEnvironment(BaseEnvironment):
    """
    ToDo: Redo docstring
    Environment that provides the XOR function imitation problem to the TFNE framework in a way that assumes that only
    the genome model's topology will be evaluated. In order to reasonably determine the genomes fitness based on the
    topology will this environment first train the weights of the model according to the neuroevolution configuration
    and then assign a fitness score to the genome.
    """

    def __init__(self, config):
        self.logger = tf.get_logger()

        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

        self.input_shape = ast.literal_eval(config.get('ENVIRONMENT', 'input_shape', fallback='(2,)'))
        self.num_output = config.getint('ENVIRONMENT', 'num_output', fallback=1)
        self.learning_rate = config.getfloat('ENVIRONMENT', 'learning_rate')
        self.epochs = config.getint('ENVIRONMENT', 'epochs')
        self.early_stop = config.getboolean('ENVIRONMENT', 'early_stop')
        if self.early_stop:
            self.early_stop_min_delta = config.getfloat('ENVIRONMENT', 'early_stop_min_delta')
            self.early_stop_patience = config.getint('ENVIRONMENT', 'early_stop_patience')

        self.logger.debug("Environment read from config: input_shape = {}".format(self.input_shape))
        self.logger.debug("Environment read from config: num_output = {}".format(self.num_output))
        self.logger.debug("Environment read from config: learning_rate = {}".format(self.learning_rate))
        self.logger.debug("Environment read from config: epochs = {}".format(self.epochs))
        self.logger.debug("Environment read from config: early_stop = {}".format(self.early_stop))
        if self.early_stop:
            self.logger.debug("Environment read from config: early_stop_min_delta = {}"
                              .format(self.early_stop_min_delta))
            self.logger.debug("Environment read from config: early_stop_patience = {}".format(self.early_stop_patience))

    def eval_genome_fitness(self, genome):
        """
        ToDo: Redo docstring
        Evaluate the genome by grabbing its model and training its weights with SGD. If activated will the training
        process stop early if no progress will take place. The genomes fitness is then determined as the inverted loss
        of the model, multiplied by 100 to get the percentage, effectively representing the model's accuracy in
        predicting the correct values.
        """
        # Get the phenotype model from the genome and declare the optimizer and loss_function
        model = genome.get_phenotype_model()
        optimizer = tf.keras.optimizers.SGD(lr=self.learning_rate)
        loss_function = tf.keras.losses.BinaryCrossentropy()

        # Compile and train the model
        model.compile(optimizer=optimizer, loss=loss_function)
        if self.early_stop:
            early_stop = tf.keras.callbacks.EarlyStopping(monitor='loss', min_delta=self.early_stop_min_delta,
                                                          patience=self.early_stop_patience)
            model.fit(self.x, self.y, batch_size=1, epochs=self.epochs, verbose=0, callbacks=[early_stop])
        else:
            model.fit(self.x, self.y, batch_size=1, epochs=self.epochs, verbose=0)

        # Calculate the genome fitness as the percentage of accuracy in its prediction, rounded to 3 decimal points
        evaluated_fitness = float(100 * (1 - loss_function(self.y, model.predict(self.x))))
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
