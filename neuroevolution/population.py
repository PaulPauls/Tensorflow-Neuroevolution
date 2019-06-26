import tensorflow as tf


class Population:
    """
    ToDo
    """
    def __init__(self):
        """
        ToDo
        """
        self.logger = tf.get_logger()

        self.genome_list = []
        self.generation_counter = None
        self.initialized_flag = False

    def add_genome(self, genome):
        """
        ToDo
        :param genome:
        :return:
        """
        self.genome_list.append(genome)

    def get_genome_list(self):
        """
        ToDo
        :return:
        """
        return self.genome_list

    def get_best_genome(self):
        """
        ToDo
        :return:
        """
        return max(self.genome_list, key=lambda x: x.fitness)

    def set_initialized(self):
        """
        ToDo
        :return:
        """
        self.generation_counter = 0
        self.initialized_flag = True

    def save_population(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("save_population() not yet implemented")

    def load_population(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("load_population() not yet implemented")
