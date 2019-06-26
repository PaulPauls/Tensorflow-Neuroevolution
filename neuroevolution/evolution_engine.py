import tensorflow as tf


class EvolutionEngine:
    """
    ToDo
    """
    def __init__(self, ne_algorithm, population, environment, config, batch_size=None):
        """
        ToDo
        :param ne_algorithm:
        :param population:
        :param environment:
        :param config:
        :param batch_size: If parameter not specified attempt training process to the maximum parallel degree.
        """
        self.logger = tf.get_logger()

        self.ne_algorithm = ne_algorithm
        self.population = population
        self.environment = environment

        if batch_size is None:
            # Determine self.batch_size
            pass
        else:
            self.batch_size = batch_size

        # Read in config parameters for evolution process
        self.max_generations_config = int(config.get('EvolutionEngine','max_generations'))
        self.fitness_threshold_config = float(config.get('EvolutionEngine','fitness_threshold'))

    def train(self, max_generations=None, fitness_threshold=None):
        """
        ToDo
        :param max_generations:
        :param fitness_threshold:
        :return:
        """

        if max_generations is None:
            max_generations = self.max_generations_config
        if fitness_threshold is None:
            fitness_threshold = self.fitness_threshold_config

        if self.population.initialized_flag is False:
            self.ne_algorithm.create_initial_population()

        while True:  # Each loop represents one complete generation in the evolution process

            # Print/Log information about current generation

            # Evaluate population
            for genome in self.population.genome_list:
                genome.fitness = self.environment.eval_genome_fitness(genome)

            # Apply neuroevolution methods to change up population and create new generation
            self.ne_algorithm.create_new_generation()

            # Break if: max_generations reached, fitness_threshold reached or population extinct.
            self.population.generation_counter += 1
            if (self.population.generation_counter == max_generations) or \
                    self.ne_algorithm.population.get_best_genome().fitness == fitness_threshold or \
                    self.ne_algorithm.check_population_extinction():
                break

        return self.ne_algorithm.population.get_best_genome()

    def _evaluate_population(self, population):
        """
        ToDo
        :param population:
        :return:
        """
        # Apply self.environment.eval_genome_fitness to whole population
