import os
from absl import logging


class EvolutionEngine:
    """
    Implementation of the central engine driving the central aspects of Neuroevolution within the train loop:
    Evaluation, Selection and Evolution. The train loop will also take care of additional tasks such as taking snapshots
    of the best genomes in each generation or regularly checking for exit loop conditions.
    """

    def __init__(self, population, environment, config, batch_size=None):
        self.population = population
        self.environment = environment

        if batch_size is None:
            # ToDo: Determine batch_size to potentially accelerate training
            pass
        else:
            self.batch_size = batch_size

        # Read in config parameters for the evolution engine
        self.max_generations_config = config.getint('EVOLUTION_ENGINE', 'max_generations')
        self.fitness_threshold_config = config.getfloat('EVOLUTION_ENGINE', 'fitness_threshold')

        logging.debug("EvolutionEngine read from config: max_generations_config = {}"
                      .format(self.max_generations_config))
        logging.debug("EvolutionEngine read from config: fitness_threshold_config = {}"
                      .format(self.fitness_threshold_config))

    def train(self, max_generations=None, fitness_threshold=None, render_best_genome_each_gen=False, render_dir=None):
        """
        Train the in the constructor specified population of genomes on the in the constructor specified environment.
        If no pretrained population has been supplied will this function initialize the population and train it in a
        loop, evolving it, evaluating it, printing a summary and possibly saving a render of the best genome of the
        current generation. The training loop will end if the specified maximum number of generations is reached,
        the best genome crosses the fitness_threshold or the population went extinct.
        """
        max_generations = self.max_generations_config if max_generations is None else max_generations
        fitness_threshold = self.fitness_threshold_config if fitness_threshold is None else fitness_threshold
        logging.info("EvolutionEngine will train population for {} generations or until fitness threshold of {} "
                     "is reached".format(max_generations, fitness_threshold))

        # Determine directory for renders of the best genome if not supplied
        if render_best_genome_each_gen and render_dir is None:
            render_dir_nr = 1
            while os.path.isdir("best_genome_each_generation_-_run_{}".format(render_dir_nr)):
                render_dir_nr += 1
            render_dir_name = "best_genome_each_generation_-_run_{}".format(render_dir_nr)
            os.mkdir(render_dir_name)
            render_dir = os.path.abspath(render_dir_name)
            logging.info("Directory for rendered best genomes of each gen: {}".format(render_dir))

        # If population not yet initialized, do so. This is unnecessary if an already evolved population is supplied.
        if self.population.get_generation_counter() is None:
            # Determine and supply the parameters for the input and output layers when initially creating genomes
            input_shape = self.environment.get_input_shape()
            num_output = self.environment.get_num_output()
            logging.info("Initializing new population...")
            self.population.initialize(input_shape, num_output)
        else:
            logging.info("Evolving already initialized population. Initial state of the population:")
            self.population.summary()

        # Create an initial evaluation before entering the loop to check if exit conditions already met and summarize it
        genome_evaluation_function = self.environment.eval_genome_fitness
        self.population.evaluate(genome_evaluation_function)
        logging.info("Summary of the population after the initial evaluation:")
        self.population.summary()

        # Evaluate and evolve population in possibly endless loop. Exit conditions:
        # population not extinct, max_generations not reached, best genome's fitness below fitness_threshold
        while not self.population.check_extinction() and \
                self.population.get_generation_counter() < max_generations and \
                self.population.get_best_genome().get_fitness() < fitness_threshold:

            # Create the next generation of the population by evolving it
            self.population.evolve()

            # Evaluate population and assign each genome a fitness score
            self.population.evaluate(genome_evaluation_function)

            # Give summary of population after each evaluation and render and save the best genome if required
            self.population.summary()
            if render_best_genome_each_gen:
                best_genome = self.population.get_best_genome()
                genome_fname = "graph_genome_{}_from_gen_{}".format(
                    best_genome.get_id(), self.population.get_generation_counter())
                best_genome.visualize(filename=genome_fname, directory=render_dir, view=False)

        best_genome = self.population.get_best_genome() if not self.population.check_extinction() else None
        return best_genome
