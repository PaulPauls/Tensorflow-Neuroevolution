import tensorflow as tf


class EvolutionEngine:
    """
    ToDo
    """
    def __init__(self, ne_algorithm, config, environment, batch_size=None):
        """
        ToDo
        :param ne_algorithm:
        :param config:
        :param environment:
        :param batch_size: If parameter not specified attempt training process to the maximum parallel degree.
        """
        self.logger = tf.get_logger()

        self.ne_algorithm = ne_algorithm
        self.config = config
        self.environment = environment

        if batch_size is None:
            # Determine self.batch_size
            pass
        else:
            self.batch_size = batch_size

    def train(self, max_generations=None, fitness_threshold=None):
        """
        ToDo
        :param max_generations:
        :param fitness_threshold:
        :return:
        """

        if self.ne_algorithm.population.initialized_flag is False:
            self.ne_algorithm.create_initial_population()

        while True:  # Each loop represents one complete generation in the evolution process

            # Print/Log information about current generation

            self._evaluate_population(self.ne_algorithm.population)

            # Select, Recombine and Mutate Population
            self.ne_algorithm.select_genomes()
            self.ne_algorithm.recombine_genomes()
            self.ne_algorithm.mutate_genomes()

            # Break if: max_generations reached, fitness_threshold reached or population extinct.
            # Otherwise loop indefinitely

            break

        pass

    def _evaluate_population(self, population):
        """
        ToDo
        :return:
        """
        # Apply self.environment.eval_genome_fitness to whole population
        for genome in population.genome_list:
            genome.fitness = self.environment.eval_genome_fitness(genome)
