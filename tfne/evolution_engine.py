import os
from datetime import datetime

import ray
import tensorflow as tf

from .encodings.base_genome import BaseGenome


class EvolutionEngine:
    """
    Central engine that drives the evolution by calling the evolutionary steps of the evolutionary algorithm in the
    correct order while checking if the evolutionary process went extinct, reached its maximum fitness/generations or
    is otherwise in order.
    """

    def __init__(self,
                 ne_algorithm,
                 environment,
                 backup_dir_path,
                 max_generations=None,
                 max_fitness=None):
        """
        Creates a new evolutionary engine instance that initializes the multiprocessing library, creates TFNE state
        backup directories and sets up evaluation environments in order to prepare for the eventual training process.
        @param ne_algorithm: instance of an TFNE compliant NE algorithm
        @param environment: instance of an TFNE compliant evaluation environment
        @param backup_dir_path: string of a directory path into which the TFNE state backups are saved
        @param max_generations: integer specifying the maximum number of generations the population should be evolved to
        @param max_fitness: float specifying the fitness of the best genome at which point the evolution process should
                            be preemptively stopped
        """
        # Register parameters and print evolution information
        self.ne_algorithm = ne_algorithm
        self.environment = environment
        self.max_generations = max_generations
        self.max_fitness = max_fitness
        print("Using Neuroevolution algorithm: {}".format(ne_algorithm.__class__.__name__))
        print("Using evaluation environment: {}".format(environment.__class__.__name__))
        print("Maximum number of generations to evolve the population: {}".format(max_generations))
        print("Maximum fitness value to evolve population up to: {}".format(max_fitness))

        # Check if NE algorithm supports multiple evaluation instances and if so, duplicate the evaluation environment
        # for parallel evaluation.
        eval_instance_count = ne_algorithm.get_eval_instance_count()
        if eval_instance_count > 1:
            self.environment = [environment]
            for _ in range(eval_instance_count - 1):
                self.environment.append(environment.duplicate())

        # Create the directory into wich the training process will backup the population each generation
        self.backup_dir_path = os.path.abspath(backup_dir_path)
        if self.backup_dir_path[-1] != '/':
            self.backup_dir_path += '/'
        backup_dir_str = datetime.now(tz=datetime.now().astimezone().tzinfo)
        backup_dir_str = backup_dir_str.strftime("tfne_state_backup_%Y-%b-%d_%H-%M-%S/")
        self.backup_dir_path += backup_dir_str
        os.makedirs(self.backup_dir_path, exist_ok=True)
        print("Creating TFNE generational Backups to directory: {}".format(self.backup_dir_path))

    def train(self) -> BaseGenome:
        """
        Starts the configured evolutionary training process. Initializes, then evaluates, summarizes and evolves
        population in loop until exit condition (extinction, max generations/fitness reached) is met. Returns the genome
        with the best achieved fitness
        @return: TFNE compliant genome with the best achieved fitness
        """
        # Initialize population. If pre-evolved population was supplied will it be used as the initial population.
        # Pass the environment as a parameter to the initialization in order to allow the determination of constraints
        # the created genomes have to adhere to.
        self.ne_algorithm.initialize_population(self.environment)

        # Start possibly endless training loop, only exited if population goes extinct, the maximum number of
        # generations or the maximum fitness has been reached
        while True:
            # Evaluate and summarize population
            generation_counter, best_fitness = self.ne_algorithm.evaluate_population(self.environment)
            self.ne_algorithm.summarize_population()

            # Backup population
            self.ne_algorithm.save_state(save_dir_path=self.backup_dir_path)

            # Exit training loop if maximum number of generations or maximum fitness has been reached
            if self.max_fitness is not None and best_fitness >= self.max_fitness:
                print("Population's best genome reached specified fitness threshold.\n"
                      "Exiting evolutionary training loop...")
                break
            if self.max_generations is not None and generation_counter >= self.max_generations:
                print("Population reached specified maximum number of generations.\n"
                      "Exiting evolutionary training loop...")
                break

            # Evolve population
            population_extinct = self.ne_algorithm.evolve_population()

            # Exit training loop if population went extinct
            if population_extinct:
                print("Population went extinct.\n"
                      "Exiting evolutionary training loop...")
                break

            # Reset models, counters, layers, etc including in the GPU to avoid memory clutter from old models as most
            # likely only limited gpu memory is available
            tf.keras.backend.clear_session()

        # Shutdown multiprocessing libraries now that training is ending
        ray.shutdown()

        # Get best genome from evolutionary process and return it. This should return the best genome of the
        # evolutionary process, even if the population went extinct.
        return self.ne_algorithm.get_best_genome()

    def get_backup_dir_path(self) -> str:
        """"""
        return self.backup_dir_path
