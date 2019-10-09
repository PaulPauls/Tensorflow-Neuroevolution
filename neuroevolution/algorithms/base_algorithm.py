from abc import ABCMeta, abstractmethod


class BaseNeuroevolutionAlgorithm(object, metaclass=ABCMeta):

    @abstractmethod
    def initialize_population(self, population, initial_pop_size, input_shape, num_output):
        """
        Initialize the population according the algorithms specifications to the size 'initial pop size'. The phenotypes
        of the genomes should accept inputs of the shape 'input_shape' and have 'num_output' nodes in their output layer
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        :param initial_pop_size: int, amount of genomes to be initialized and added to the population
        :param input_shape: tuple, shape of the input vector for the NN model to be created
        :param num_output: int, number of nodes in the output layer of the NN model to be created
        """
        raise NotImplementedError("Should implement initialize_population()")

    @abstractmethod
    def evolve_population(self, population, pop_size_fixed):
        """
        Evolve the population according to the algorithms specifications.
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        :param pop_size_fixed: bool flag, indicating if the size of the population can be different after the evolution
                               of the current generation is complete
        """
        raise NotImplementedError("Should implement evolve_population()")

    @abstractmethod
    def evaluate_population(self, population, genome_eval_function):
        """
        Evaluate the population according to the algorithms specification by using the callable method
        'genome_eval_function', which is intended to return the evaluated fitness for each supplied genome.
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        :param genome_eval_function: callable method that takes a genome as input and returns the fitness score
                                     corresponding to the genomes performance in the environment
        """
        raise NotImplementedError("Should implement evaluate_population()")

    @abstractmethod
    def summarize_population(self, population):
        """
        Output a summary of the population, giving a concise overview of the status of the population.
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        """
        raise NotImplementedError("Should implement summarize_population()")
