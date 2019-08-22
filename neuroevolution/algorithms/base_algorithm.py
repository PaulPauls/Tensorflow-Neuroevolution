from abc import ABCMeta, abstractmethod


class BaseNeuroevolutionAlgorithm(object, metaclass=ABCMeta):
    """
    Base class for potential neuroevolution algorithms to subclass. This ensures that the algorithms used in the
    Tensorflow-Neuroevolution framework implement the required functions in the intended way.
    """

    @abstractmethod
    def create_initial_genome(self, input_shape, num_output):
        raise NotImplementedError("Should implement create_initial_genome()")

    @abstractmethod
    def create_new_generation(self, population):
        raise NotImplementedError("Should implement create_new_generation()")
