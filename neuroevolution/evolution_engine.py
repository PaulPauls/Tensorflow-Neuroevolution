from absl import logging


class EvolutionEngine:
    def __init__(self, population, environment):
        self.population = population
        self.environment = environment

    def train(self, max_generations=None, fitness_threshold=None, genome_render_agent=None, pop_backup_agent=None):
        # Log if max_generations, fitness_threshold or reporting agents are supplied and what that means
        self._log_parameters(max_generations, fitness_threshold, genome_render_agent, pop_backup_agent)

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

        while self._check_exit_conditions(max_generations, fitness_threshold):
            # Create the next generation of the population by evolving it
            self.population.evolve()

            # Evaluate population and assign each genome a fitness score
            self.population.evaluate(genome_eval_function)

            # Speciate population if NE algorithm supports it
            self.population.speciate()

            # Give summary of population after each evaluation and call the reporting agents if supplied
            self.population.summary()
            if genome_render_agent is not None:
                genome_render_agent(self.population)
            if pop_backup_agent is not None:
                pop_backup_agent(self.population)

        best_genome = self.population.get_best_genome() if not self.population.check_extinction() else None
        return best_genome

    def _check_exit_conditions(self, max_generations, fitness_threshold):
        if self.population.check_extinction():
            logging.info("Population extinct. Exiting evolutionary training loop...")
            return False
        if max_generations is not None and self.population.get_generation_counter() >= max_generations:
            logging.info("Population reached specified maximum number of generations. "
                         "Exiting evolutionary training loop...")
            return False
        if fitness_threshold is not None and self.population.get_best_genome().get_fitness >= fitness_threshold:
            logging.info("Population's best genome reached specified fitness threshold. "
                         "Exiting evolutionary training loop...")
            return False
        return True

    @staticmethod
    def _log_parameters(max_generations, fitness_threshold, genome_render_agent, pop_backup_agent):
        if max_generations is not None and fitness_threshold is not None:
            logging.info("Evolution will progress for a maximum of {} generations or until a fitness of {} is reached"
                         .format(max_generations, fitness_threshold))
        elif max_generations is not None:
            logging.info("Evolution will progress for a maximum of {} generations".format(max_generations))
        elif fitness_threshold is not None:
            logging.info("Evolution will progress until a fitness of {} is reached".format(fitness_threshold))
        else:
            logging.info("Evolution will progress indefinitely")
        if genome_render_agent is not None:
            if genome_render_agent.view:
                logging.info("Evolution will show the graph of the best genome after each generation and save the "
                             "rendering to the directory '{}'".format(genome_render_agent.render_dir_path))
            else:
                logging.info("Evolution will _not_ show the graph of the best genome after each generation but save "
                             "the rendering to the directory '{}'".format(genome_render_agent.render_dir_path))
        if pop_backup_agent is not None:
            logging.info("Evolution will backup the population every {} generations to the directory '{}'"
                         .format(pop_backup_agent.backup_periodicity, pop_backup_agent.backup_dir_path))
