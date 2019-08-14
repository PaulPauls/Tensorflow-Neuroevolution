from abc import ABCMeta, abstractmethod


class BaseNeuroevolutionAlgorithm(object, metaclass=ABCMeta):

    @abstractmethod
    def create_initial_genome(self):
        raise NotImplementedError("Should implement create_initial_genome()")

    @abstractmethod
    def create_new_generation(self):
        raise NotImplementedError("Should implement create_new_generation()")
