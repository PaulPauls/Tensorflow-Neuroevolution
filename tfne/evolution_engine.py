import os
from datetime import datetime

import ray
import tensorflow as tf
from absl import logging

from .encodings.base_genome import BaseGenome


class EvolutionEngine:
    """"""

    def __init__(self,
                 ne_algorithm,
                 backup_dir_path,
                 num_cpus=None,
                 num_gpus=None,
                 max_generations=None,
                 max_fitness=None):
        """"""
        # Register parameters
        self.ne_algorithm = ne_algorithm
        self.max_generations = max_generations
        self.max_fitness = max_fitness
        print("Using Neuroevolution algorithm: {}".format(ne_algorithm.__class__.__name__))
        print("Maximum number of generations to evolve the population: {}".format(max_generations))
        print("Maximum fitness value to evolve population up to: {}".format(max_fitness))

        # Initiate the Multiprocessing library ray and determine the actually available CPUs and GPUs as well as the
        # desired verbosity level for the genome evaluation
        ray.init(num_cpus=num_cpus, num_gpus=num_gpus)
        self.available_num_cpus = ray.available_resources()['CPU']
        self.available_num_gpus = len(ray.get_gpu_ids())
        self.verbosity = 0 if not logging.level_debug() else 1
        print("Initialized the ray library with {} CPUs and {} GPUs".format(self.available_num_cpus,
                                                                            self.available_num_gpus))

        # Initialize the environments through the registered algorithm with the determined parameters for parallel
        # evaluation and verbosity
        self.ne_algorithm.initialize_environments(num_cpus=self.available_num_cpus,
                                                  num_gpus=self.available_num_gpus,
                                                  verbosity=self.verbosity)

        # Create the directory into wich the training process will backup the population each generation
        self.backup_dir_path = os.path.abspath(backup_dir_path)
        if self.backup_dir_path[-1] != '/':
            self.backup_dir_path += '/'
        self.backup_dir_path += datetime.now(tz=datetime.now().astimezone().tzinfo).strftime("run_%Y-%b-%d_%H-%M-%S/")
        os.makedirs(self.backup_dir_path)
        print("Backing up population to directory: {}".format(self.backup_dir_path))

    def train(self) -> BaseGenome:
        """"""
        # Initialize population. If pre-evolved population was supplied will this be used as the initial population.
        self.ne_algorithm.initialize_population()

        # Start possibly endless training loop, only exited if population goes extinct, the maximum number of
        # generations or the maximum fitness has been reached
        while True:
            # Evaluate and summarize population
            generation_counter, best_fitness = self.ne_algorithm.evaluate_population()
            self.ne_algorithm.summarize_population()

            # Backup population in according gen directory
            gen_backup_dir_path = self.backup_dir_path + f"gen_{generation_counter}/"
            os.makedirs(gen_backup_dir_path)
            self.ne_algorithm.save_population(save_dir_path=gen_backup_dir_path)

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

            # Reset models, counters, layers, etc including in the GPU to avoid clutter from old models as most likely
            # only limited memory is available
            tf.keras.backend.clear_session()

        # Determine best genome from evolutionary process and return it. This should return the best genome of the
        # evolutionary process, even if the population went extinct.
        return self.ne_algorithm.get_best_genome()
