from absl import logging


class EvolutionEngine:
    def __init__(self, population, environment):
        self.population = population
        self.environment = environment

    def train(self, max_generations=None, fitness_threshold=None, genome_render_agent=None, pop_backup_agent=None):

        # Log if max_generations, fitness_threshold or reporting agents are supplied and what that means
        if max_generations is not None and fitness_threshold is not None:
            logging.info("TODO")
        elif max_generations is not None:
            logging.info("TODO")
        elif fitness_threshold is not None:
            logging.info("TODO")
        else:
            logging.info("TODO")
        if genome_render_agent is not None:
            logging.info("TODO")
        if pop_backup_agent is not None:
            logging.info("TODO")

        # If population not yet initialized, do so. This is unnecessary if an already evolved population is supplied.
        if self.population.get_generation_counter() is None:
            # Determine and supply the parameters for the input and output layers when initially creating genomes
            input_shape = self.environment.get_input_shape()
            num_output = self.environment.get_num_output()
            logging.info("Initializing new population...")
            self.population.initialize(input_shape, num_output)
        else:
            logging.info("Evolving pre-loaded population. Initial state of the population:")
            self.population.summary()

        # Declare the genome evaluation function from the environment
        genome_eval_function = self.environment.eval_genome_fitness

        # Create an initial evaluation before entering the loop to check if exit conditions already met and summarize it
        self.population.evaluate(genome_eval_function)
        logging.info("Summary of the population after the initial evaluation:")
        self.population.summary()

        while not self._check_exit_conditions(max_generations, fitness_threshold):
            # Create the next generation of the population by evolving it
            self.population.evolve()

            # Evaluate population and assign each genome a fitness score
            self.population.evaluate(genome_eval_function)

            # Give summary of population after each evaluation and call the reporting agents if supplied
            self.population.summary()
            if genome_render_agent is not None:
                genome_render_agent(self.population)
            if pop_backup_agent is not None:
                pop_backup_agent(self.population)

        best_genome = self.population.get_best_genome() if not self.population.check_extinction() else None
        return best_genome


    def _check_exit_conditions(self, max_generations, fitness_threshold):
        '''
        self.population.check_extinction() and \
        self.population.get_generation_counter() < max_generations and \
        self.population.get_best_genome().get_fitness() < fitness_threshold
        '''
        pass
