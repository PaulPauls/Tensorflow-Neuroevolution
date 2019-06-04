class EvolutionEngine:
    """
    ToDo
    """
    def __init__(self, config, population, eval_genome_fitness_function, batch_size=None):
        """
        ToDo
        :param config:
        :param population:
        :param eval_genome_fitness_function:
        :param batch_size: If parameter not specified attempt training process to the maximum parallel degree.
        """
        pass

    def train(self, max_generations):
        """
        ToDo
        :param max_generations:
        :return:
        """
        for _ in range(max_generations):
            # Print info about generation start

            # Apply eval_genome_fitness_function to the parallel degree specified in batch_size to the whole population

            # Print info about trained generation

            # Do housework (determine best_genome, determine if fitness_threshold met, determine if max_generations,
            # determine if population extinct)
            pass
        pass

    def set_verbosity(self, verbosity):
        """
        ToDo
        ToDo: Alternatively replace this verbosity setting with the addition of Progress Reporters
        :param verbosity:
        :return:
        """
        pass
