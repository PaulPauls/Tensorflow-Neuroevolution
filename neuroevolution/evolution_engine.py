from absl import logging

from .encodings.base_genome import BaseGenome


class EvolutionEngine:
    """
    Implementation of the engine driving the central aspects of Neuroevolution - Evaluation, Selection and Evolution -
    through a central training loop that also takes care of additional tasks such as logging, backups, etc. The
    neuroevolutionary process judges genomes on the constructor supplied environment and evolves the constructor
    supplied population through the neuroevolutionary algorithm it is configured with.
    """

    def __init__(self, population, environment):
        self.population = population
        self.environment = environment
        self._log_class_parameters()

    def _log_class_parameters(self):
        logging.debug("EvolutionEngine parameter: environment = {}".format(self.environment.__class__.__name__))

    def train(self, max_generations=None, fitness_threshold=None, reporting_agents=None) -> BaseGenome:
        """
        Train the in the constructor specified population of genomes on the specified environment through a
        neuroevolutionary process of continuous evolution and evaluation. If no pre-evolved population is supplied in
        the constructor will this training loop first initialize the population. The evolutionary progress will then
        progress indefinitely or until one of the exit conditions specified as this functions parameter is met,
        whereupon the populations best genome (in terms of achieved fitness) is returned. The evolutionary progress will
        be regularly backed up, visualized or reported upon by the specified list of reporting_agents.
        :param max_generations: int parameter specifying the maximum number of generations a population should be
                                evolved before the evolutionary process should be aborted.
        :param fitness_threshold: float parameter specifying the minimum fitness the population's best genome should
                                  possess before the evolutionary process should be aborted.
        :param reporting_agents: list of ReportingAgents, intended to log the evolutionary process by regularly saving/
                                 visualizing the populations best genome, saving/visualizing the speciation, backing up
                                 the population, etc.
        :return: The populations best genome (in terms of achieved fitness) resulting from the evolutionary process
        """
        # Log if max_generations, fitness_threshold or reporting agents are supplied and their respective effects
        self._log_train_parameters(max_generations, fitness_threshold, reporting_agents)

        # If population not yet initialized, do so. This is unnecessary if an already evolved population is supplied.
        if self.population.get_generation_counter() is None:
            # Determine and supply the parameters for the input and output layers when initially creating genomes
            input_shape = self.environment.get_input_shape()
            num_output = self.environment.get_num_output()
            self.population.initialize(input_shape, num_output)
        else:
            logging.info("Evolving preloaded population. Initial state of the population:")
            self.population.summary()

        # Declare the environment and its genome evaluation function
        environment_name = self.environment.__class__.__name__
        genome_eval_function = self.environment.eval_genome_fitness

        # Create an initial evaluation and summary before entering the loop, to check if exit conditions already met
        self.population.evaluate(environment_name, genome_eval_function)
        logging.info("Summarizing population after initial evaluation...")
        self.population.summary()

        reporting_agents_supplied = reporting_agents is not None
        while self._check_exit_conditions(max_generations, fitness_threshold):
            # Create the next generation of the population by evolving it
            self.population.evolve()

            # Evaluate population and assign each genome a fitness score
            self.population.evaluate(environment_name, genome_eval_function)

            # Give summary of population after each evaluation
            self.population.summary()

            # Call reporting agents if supplied
            if reporting_agents_supplied:
                for reporting_agent in reporting_agents:
                    reporting_agent(self.population)

        # Determine best genome resulting from the evolution and then return it, after the process came to an end
        best_genome = self.population.get_best_genome() if not self.population.check_extinction() else None
        return best_genome

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

    def _check_exit_conditions(self, max_generations, fitness_threshold) -> bool:
        """
        Checks the exit conditions of the evolutionary training loop and returns False if one exit condition is met,
        otherwise True. Exit conditions are met if population is extinct, the specified maximum number of generations
        is reached or the fitness of the populations best genome surpasses the fitness threshold.
        :param max_generations: int parameter specifying the maximum number of generations a population should be
                                evolved before the evolutionary process should be aborted.
        :param fitness_threshold: float parameter specifying the minimum fitness the population's best genome should
                                  possess before the evolutionary process should be aborted.
        :return: False if an exit condition is met and the Training loop should not continue. True if no exit condition
                 is met and the training loop should continue.
        """
        if self.population.check_extinction():
            logging.info("Population extinct. Exiting evolutionary training loop...")
            return False
        if max_generations is not None and self.population.get_generation_counter() >= max_generations:
            logging.info("Population reached specified maximum number of generations. "
                         "Exiting evolutionary training loop...")
            return False
        if fitness_threshold is not None and self.population.get_best_genome().get_fitness() >= fitness_threshold:
            logging.info("Population's best genome reached specified fitness threshold. "
                         "Exiting evolutionary training loop...")
            return False
        return True
