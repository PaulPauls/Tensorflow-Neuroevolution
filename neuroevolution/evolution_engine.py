from absl import logging


class EvolutionEngine:
    def __init__(self, population, environment):
        self.population = population
        self.environment = environment
        self._log_class_parameters()

    def _log_class_parameters(self):
        logging.debug("EvolutionEngine parameter: environment = {}".format(self.environment.__class__.__name__))

    def train(self, max_generations=None, fitness_threshold=None, reporting_agents=None):
        # Log if max_generations, fitness_threshold or reporting agents are supplied and what that means
        self._log_train_parameters(max_generations, fitness_threshold, reporting_agents)

        # If population not yet initialized, do so. This is unnecessary if an already evolved population is supplied.
        if self.population.get_generation_counter() is None:
            # Determine and supply the parameters for the input and output layers when initially creating genomes
            input_shape = self.environment.get_input_shape()
            num_output = self.environment.get_num_output()
            self.population.initialize(input_shape, num_output)
        else:
            logging.info("Evolving pre-loaded population. Initial state of the population:")
            self.population.summary()

        # Declare the genome evaluation function from the environment
        environment_name = self.environment.__class__.__name__
        genome_eval_function = self.environment.eval_genome_fitness
        reporting_agents_present = reporting_agents is not None

        # Create an initial evaluation before entering the loop to check if exit conditions already met and summarize it
        self.population.evaluate(environment_name, genome_eval_function)
        logging.info("Summarizing population after initial evaluation...")
        self.population.summary()

        while self._check_exit_conditions(max_generations, fitness_threshold):
            # Create the next generation of the population by evolving it
            self.population.evolve()

            # Evaluate population and assign each genome a fitness score
            self.population.evaluate(environment_name, genome_eval_function)

            # Give summary of population after each evaluation and call the reporting agents if supplied
            self.population.summary()

            # Call Reporting Agents if supplied
            if reporting_agents_present:
                for reporting_agent in reporting_agents:
                    reporting_agent(self.population)

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
    def _log_train_parameters(max_generations, fitness_threshold, reporting_agents):
        if max_generations is not None and fitness_threshold is not None:
            logging.info("Evolution will progress for a maximum of {} generations or until a fitness of {} is reached"
                         .format(max_generations, fitness_threshold))
        elif max_generations is not None:
            logging.info("Evolution will progress for a maximum of {} generations".format(max_generations))
        elif fitness_threshold is not None:
            logging.info("Evolution will progress until a fitness of {} is reached".format(fitness_threshold))
        else:
            logging.info("Evolution will progress indefinitely")

        if reporting_agents is not None:
            for reporting_agent in reporting_agents:
                reporting_agent.log_parameters()
