from abc import ABCMeta, abstractmethod


class BaseEnvironment(object, metaclass=ABCMeta):
    """
    Base class for potential neuroevolution environments to subclass. This ensures that the environments used in the
    Tensorflow-Neuroevolution framework implement the required functions in the intended way.
    """

    @abstractmethod
    def eval_genome_fitness(self, genome):
        raise NotImplementedError("Should implement eval_genome_fitness()")

    @abstractmethod
    def replay_genome(self, genome):
        raise NotImplementedError("Should implement replay_genome()")

    @abstractmethod
    def get_input_shape(self):
        raise NotImplementedError("Should implement get_input_shape()")

    @abstractmethod
    def get_num_output(self):
        raise NotImplementedError("Should implement get_num_output()")
