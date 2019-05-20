class Population:
    """
    ToDo
    """
    def __init__(self, config, existing_population=None):
        """
        ToDo
        :param config:
        :param existing_population:
        """
        self.ne_algorithm = config.ne_algorithm

        if existing_population is None:
            self.population = self._create_initial_population()
        else:
            self.population = existing_population

        pass

    def _create_initial_population(self):
        """
        ToDo
        :return:
        """
        self.ne_algorithm.create_genomes()
        self.ne_algorithm.speciate_genomes()
        pass

    def _evolve_population(self):
        """
        ToDo
        :return:
        """
        self.ne_algorithm.select_genomes()
        self.ne_algorithm.mutate_genomes()
        self.ne_algorithm.speciate_genomes()
        pass

    def get_best_genome(self):
        """
        ToDo
        :return:
        """
        pass

    def save_population(self):
        """
        ToDo
        :return:
        """
        pass

    def load_population(self):
        """
        ToDo
        :return:
        """
        pass
