import tensorflow as tf


class EvolutionEngine:
    def __init__(self, population, environment, config, batch_size=None):
        self.logger = tf.get_logger()

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
        max_generations = self.max_generations_config if max_generations is None else max_generations
        fitness_threshold = self.fitness_threshold_config if fitness_threshold is None else fitness_threshold

        # If population not yet initialized, do so. This is unnecessary if an already evolved population is supplied.
        if self.population.get_generation_counter() is None:
            # Determine and supply the parameters for the input and output layers when initially creating genomes
            input_shape = self.environment.get_input_shape()
            num_output = self.environment.get_num_output()
            self.population.initialize(input_shape, num_output)
        else:
            self.logger.info("Evolving already initialized population. Initial state of the population:")
            self.population.summary()

        # Evaluate and evolve population in possibly endless loop, according to loop exit conditions.
        while True:
            # Check exit conditions for loop: population extinct, max_generations reached,
            #                                 At least one generation old and fitness_threshold reached
            if self.population.check_extinction() or \
               self.population.get_generation_counter() > max_generations or \
               self.population.get_generation_counter() > 0 and \
               self.population.get_best_genome().get_fitness() >= fitness_threshold:
                break

            # Evaluate population and assign each genome a fitness score
            genome_evaluation_function = self.environment.eval_genome_fitness
            self.population.evaluate(genome_evaluation_function)

            # Give summary of population after each evaluation
            self.population.summary()

            # Create the next generation of the population
            self.population.evolve()

        best_genome = self.population.get_best_genome() if not self.population.check_extinction() else None
        return best_genome
