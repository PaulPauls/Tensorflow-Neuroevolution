from abc import ABCMeta, abstractmethod


class BaseNeuroevolutionAlgorithm(object, metaclass=ABCMeta):

    @abstractmethod
    def create_initial_genome(self, input_shape, num_output):
        raise NotImplementedError("Should implement create_initial_genome()")

    @abstractmethod
    def create_mutated_genome(self, genome):
        raise NotImplementedError("Should implement create_mutated_genome()")
