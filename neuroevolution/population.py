import tensorflow as tf


class Population:
    def __init__(self):
        self.logger = tf.get_logger()

        self.genome_list = []
        self.generation_counter = None
        self.initialized_flag = False

    def add_genome(self, genome):
        self.genome_list.append(genome)

    def remove_genome(self, genome):
        self.genome_list.remove(genome)

    def get_genome_list(self):
        return self.genome_list

    def get_genome(self, i):
        return self.genome_list[i]

    def get_best_genome(self):
        return max(self.genome_list, key=lambda x: x.get_fitness())

    def get_worst_genome(self):
        return min(self.genome_list, key=lambda x: x.get_fitness())

    def set_initialized(self):
        self.generation_counter = 0
        self.initialized_flag = True

    def increment_generation_counter(self):
        self.generation_counter += 1

    def get_generation_counter(self):
        return self.generation_counter

    def check_extinction(self):
        return len(self.genome_list) == 0

    def save_population(self):
        raise NotImplementedError("save_population() not yet implemented")

    def load_population(self):
        raise NotImplementedError("load_population() not yet implemented")
