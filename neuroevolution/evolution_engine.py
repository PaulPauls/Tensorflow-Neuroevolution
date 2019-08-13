import tensorflow as tf


class EvolutionEngine:
    def __init__(self, ne_algorithm, population, environment, config, batch_size=None):
        self.logger = tf.get_logger()

        self.ne_algorithm = ne_algorithm
        self.population = population
        self.environment = environment

        if batch_size is None:
            # ToDo: Determine self.batch_size
            pass
        else:
            self.batch_size = batch_size

        # Read in config parameters for evolution process
        self.max_generations_config = config.getint('EVOLUTION_ENGINE', 'max_generations')
        self.fitness_threshold_config = config.getfloat('EVOLUTION_ENGINE', 'fitness_threshold')

    def train(self, max_generations=None, fitness_threshold=None):
        pass
