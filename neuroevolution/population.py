import json
from absl import logging


class Population:
    def __init__(self, ne_algorithm, config):
        self.ne_algorithm = ne_algorithm
        self.initial_pop_size = None
        self.pop_size_fixed = None
        self._read_config_parameters(config)
        self._log_class_parameters()

        self.pop_size = 0
        self.generation_counter = None

        self.genomes = list()

    def _read_config_parameters(self, config):
        self.initial_pop_size = config.getint('POPULATION', 'initial_pop_size')
        self.pop_size_fixed = config.getboolean('POPULATION', 'pop_size_fixed')

    def _log_class_parameters(self):
        logging.debug("Population parameter: ne_algorithm = {}".format(self.ne_algorithm.__class__.__name__))
        logging.debug("Population read from config: initial_pop_size = {}".format(self.initial_pop_size))
        logging.debug("Population read from config: pop_size_fixed = {}".format(self.pop_size_fixed))

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

    def check_extinction(self):
        return self.pop_size == 0

    def add_genome(self, genome):
        self.genomes.append(genome)
        self.pop_size += 1

    def delete_genome(self, genome_index):
        del self.genomes[genome_index]
        self.pop_size -= 1

    def replace_population(self, new_genomes):
        del self.genomes
        self.genomes = new_genomes

    def get_genome(self, genome_index):
        return self.genomes[genome_index]

    def get_pop_size(self):
        return self.pop_size

    def get_generation_counter(self):
        return self.generation_counter

    def get_best_genome(self):
        return max(self.genomes, key=lambda x: x.get_fitness())

    def get_worst_genome(self):
        return min(self.genomes, key=lambda x: x.get_fitness())

    def get_average_fitness(self):
        fitness_sum = 0
        for genome in self.genomes:
            fitness_sum += genome.get_fitness()
        return round(fitness_sum / self.pop_size, 3)

    def save_population(self, save_file_path):
        raise NotImplementedError()

    def load_population(self, encoding, load_file_path):
        raise NotImplementedError()

    '''
    def save_population(self, save_file_path):
        serialized_genomes = {species_id: [genome.serialize() for genome in species_genomes]
                              for species_id, species_genomes in self.genomes.items()}
        serialized_population = {
            'generation_counter': self.generation_counter,
            'pop_size': self.pop_size,
            'species_count': self.species_count,
            'species_avg_fitness_log': self.species_avg_fitness_log,
            'species_best_fitness_log': self.species_best_fitness_log,
            'genomes': serialized_genomes
        }
        with open(save_file_path, 'w') as save_file:
            json.dump(serialized_population, save_file, indent=4)
        logging.info("Saved population to file '{}'".format(save_file_path))

    def load_population(self, encoding, load_file_path):
        with open(load_file_path, 'r') as load_file:
            loaded_population = json.load(load_file)
        self.generation_counter = loaded_population['generation_counter']
        self.pop_size = loaded_population['pop_size']
        self.species_count = loaded_population['species_count']
        self.species_id_counter = max(loaded_population['genomes'])
        self.species_avg_fitness_log = loaded_population['species_avg_fitness_log']
        self.species_best_fitness_log = loaded_population['species_best_fitness_log']
        assert not self.pop_size_fixed or self.pop_size == self.initial_pop_size

        self.genomes = dict()
        for species_id, species_genomes in loaded_population['genomes'].items():
            deserialized_genome_list = encoding.deserialize_genome_list(species_genomes)
            self.genomes[species_id] = deque(deserialized_genome_list)

        # Work around limitation of json serialization that only saves integer keys as strings by converting all keys
        # of the deserialized json back to integers.
        for key, value in self.species_avg_fitness_log.items():
            del self.species_avg_fitness_log[key]
            self.species_avg_fitness_log[int(key)] = value
        for key, value in self.species_best_fitness_log.items():
            del self.species_best_fitness_log[key]
            self.species_best_fitness_log[int(key)] = value
        for key, value in self.genomes.items():
            del self.genomes[key]
            self.genomes[int(key)] = value

        logging.info("Loaded population of encoding '{}' from file '{}'. Summary of the population:"
                     .format(encoding.__class__.__name__, load_file_path))
        self.summary()
    '''
