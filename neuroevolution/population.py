import json
from absl import logging

from .encodings.base_genome import BaseGenome


class Population:
    """
    Abstraction of the collection of all genomes evaluated and evolved through the TFNE framework. All genomes are saved
    in a hashtable and are accessed and added via a key, which in turn has to be remembered by the NE algorithm
    utilizing the population (though a list of all keys can be queried via 'get_genome_ids()'). Aside from the simple
    collection abstraction tasks does the population also serve as middle-man for initialize(), evolve(), evaluate(),
    etc calls, for which required maintenance and house work operations are performed before the request is forwarded
    to the NE algorthm utilized by the population.
    """

    def __init__(self, config, ne_algorithm):
        self.ne_algorithm = ne_algorithm

        # Declare, read in and log config parameters for the population
        self.initial_pop_size = None
        self.pop_size_fixed = None
        self._read_config_parameters(config)
        self._log_class_parameters()

        self.pop_size = 0
        self.generation_counter = None

        # Declare the actual container for all genomes as a dict, associating genome-ids (dict key) with their
        # respective genome (dict value)
        self.genomes = dict()

    def _read_config_parameters(self, config):
        """
        Read the class parameters supplied via the config file
        :param config: ConfigParser Object which has processed the supplied configuration
        """
        self.initial_pop_size = config.getint('POPULATION', 'initial_pop_size')
        self.pop_size_fixed = config.getboolean('POPULATION', 'pop_size_fixed')

    def _log_class_parameters(self):
        logging.debug("Population parameter: ne_algorithm = {}".format(self.ne_algorithm.__class__.__name__))
        logging.debug("Population config: initial_pop_size = {}".format(self.initial_pop_size))
        logging.debug("Population config: pop_size_fixed = {}".format(self.pop_size_fixed))

    def initialize(self, input_shape, num_output):
        logging.info("Initializing population to size {}...".format(self.initial_pop_size))
        self.generation_counter = 0
        self.ne_algorithm.initialize_population(self, self.initial_pop_size, input_shape, num_output)

    def evolve(self):
        logging.info("Evolving population of size {} from generation {}..."
                     .format(self.pop_size, self.generation_counter))
        self.generation_counter += 1
        self.ne_algorithm.evolve_population(self, self.pop_size_fixed)

    def evaluate(self, environment_name, genome_eval_function):
        logging.info("Evaluating population of size {} from generation {} on the environment '{}'..."
                     .format(self.pop_size, self.generation_counter, environment_name))
        self.ne_algorithm.evaluate_population(self, genome_eval_function)

    def summary(self):
        logging.info("Summarizing population of size {} from generation {}..."
                     .format(self.pop_size, self.generation_counter))
        self.ne_algorithm.summarize_population(self)

    def check_extinction(self) -> bool:
        return self.pop_size == 0

    def add_genome(self, genome_id, genome):
        self.genomes[genome_id] = genome
        self.pop_size += 1

    def delete_genome(self, genome_id):
        del self.genomes[genome_id]
        self.pop_size -= 1

    def get_genome_ids(self) -> []:
        return self.genomes.keys()

    def get_genome(self, genome_id) -> BaseGenome:
        return self.genomes[genome_id]

    def get_pop_size(self) -> int:
        return self.pop_size

    def get_generation_counter(self) -> int:
        return self.generation_counter

    def get_best_genome(self) -> BaseGenome:
        """
        :return: genome from the population with the best fitness score
        """
        return max(self.genomes.values(), key=lambda x: x.get_fitness())

    def get_worst_genome(self) -> BaseGenome:
        """
        :return: genome from the population with the worst fitness score
        """
        return min(self.genomes.values(), key=lambda x: x.get_fitness())

    def get_average_fitness(self) -> float:
        """
        :return: average fitness of all genomes from the population, rounded to 3 decimal places
        """
        fitness_sum = 0
        for genome in self.genomes.values():
            fitness_sum += genome.get_fitness()
        return round(fitness_sum / self.pop_size, 3)

    def save_population(self, save_file_path):
        """
        Shallow serializes and saves the population to a json file at the specified location. The serialization is
        shallow as only genome characteristics are saved that are required to recreate them from scratch - but no
        internal states (such as fitness, encoding gene counter, species assignment, etc) are serialized.
        :param save_file_path: string of file path, specifying where the serialized population should be saved
        """
        serialized_population = {
            'generation_counter': self.generation_counter,
            'pop_size': self.pop_size,
            'genomes': [genome.serialize() for genome in self.genomes.values()]
        }
        with open(save_file_path, 'w') as save_file:
            json.dump(serialized_population, save_file, indent=4)
        logging.info("Saved population to file '{}'".format(save_file_path))

    def load_population(self, encoding, load_file_path):
        raise NotImplementedError()
