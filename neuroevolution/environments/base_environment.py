"""
Base class for potential environments to subclass. This ensures that environments used in the Tensorflow-Neuroevolution
framework implement the required functions in the intended way.
"""

from abc import ABCMeta, abstractmethod


class BaseEnvironment(object):
    """
    ToDo
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def eval_genome_fitness(self, genome):
        """
        ToDo: Require subclasses to return a float/int as fitness value from this function
        ToDo: Input genome; apply the genome to the test environments; Return its calculated resulting fitness value
        :param genome:
        :return:
        """
        pass

    @abstractmethod
    def replay_genome(self, genome):
        """
        ToDo: Input genome, apply it to the test environment, though this time render the process of it being applied
        :param genome:
        :return: None
        """
        pass
