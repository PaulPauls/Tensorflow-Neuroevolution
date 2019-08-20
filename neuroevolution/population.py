from random import randint
from collections import deque

import tensorflow as tf


class Population:
    def __init__(self, ne_algorithm, config):
        self.logger = tf.get_logger()

        self.ne_algorithm = ne_algorithm

        # Read in config parameters for population
        self.replacement_percentage = config.getfloat('POPULATION', 'replacement_percentage')
        self.supplied_pop_size = config.getint('POPULATION', 'pop_size')

        # create flexible pop_size, genome container that is the actual population and set generation_counter to
        # uninitialized
        self.pop_size = self.supplied_pop_size
        self.genomes = deque(maxlen=self.supplied_pop_size)
        self.generation_counter = None

    def initialize(self, input_shape, num_output):
        self.logger.info("Initializing population of size {}".format(self.supplied_pop_size))
        for _ in range(self.supplied_pop_size):
            new_initialized_genome = self.ne_algorithm.create_initial_genome(input_shape, num_output)
            self.genomes.append(new_initialized_genome)
        self.generation_counter = 0

    def evaluate(self, genome_evaluation_function):
        # Evaluate each genome that has so far not been evaluated (effectively having a fitness_score of None)
        self.logger.debug("Evaluating {} genomes in generation {} ...".format(self.pop_size, self.generation_counter))
        for genome in self.genomes:
            if genome.get_fitness() == 0:
                genome_evaluation_function(genome)
                # self.logger.debug('Genome {} scored fitness {}'.format(genome.get_id(), genome.get_fitness()))

    def evolve(self):
        replacement_count = int(self.replacement_percentage * self.pop_size)
        # Remove the in replacement_count specified amount of the worst performing members of the population
        for _ in range(replacement_count):
            worst_genome = self.get_worst_genome()
            self.genomes.remove(worst_genome)

        # Add the same number of mutated genomes (mutated from random genomes still in pop) back to the population
        for _ in range(replacement_count):
            genome_to_mutate = self.genomes[randint(0, self.pop_size-replacement_count-1)]
            mutated_genome = self.ne_algorithm.create_mutated_genome(genome_to_mutate)
            self.genomes.append(mutated_genome)

        self.pop_size = len(self.genomes)
        self.generation_counter += 1
        self.logger.debug(
            "{} genomes have been replaced. After the evolution there are {} genomes present in generation {}"
            .format(replacement_count, self.pop_size, self.generation_counter))

    def check_extinction(self):
        return self.pop_size == 0

    def summary(self, best_genome_render_dir=None):
        best_genome = self.get_best_genome()
        best_fitness = best_genome.get_fitness()
        average_fitness = self.get_average_fitness()
        self.logger.info("#### GENERATION: {} #### BEST_FITNESS: {} #### AVERAGE_FITNESS: {} #### POP_SIZE: {} ####".
                         format(self.generation_counter, best_fitness, average_fitness, self.pop_size))
        for i in range(self.pop_size):
            self.logger.info(self.genomes[i])
        self.logger.info("#"*100 + "\n")

        if best_genome_render_dir is not None:
            filename = "graph_genome_{}_from_gen_{}".format(best_genome.get_id(), self.generation_counter)
            best_genome.visualize(filename=filename, directory=best_genome_render_dir, view=False)

    def get_genome(self, i):
        return self.genomes[i]

    def get_best_genome(self):
        return max(self.genomes, key=lambda x: x.get_fitness())

    def get_worst_genome(self):
        return min(self.genomes, key=lambda x: x.get_fitness())

    def get_generation_counter(self):
        return self.generation_counter

    def get_average_fitness(self):
        fitness_sum = sum(genome.get_fitness() for genome in self.genomes)
        average_fitness = round(fitness_sum / self.pop_size, 3)
        return average_fitness

    def load_population(self):
        raise NotImplementedError("load_population() not yet implemented")

    def save_population(self):
        raise NotImplementedError("save_population() not yet implemented")
