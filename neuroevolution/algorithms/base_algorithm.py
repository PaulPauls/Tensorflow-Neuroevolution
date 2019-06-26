from abc import ABCMeta, abstractmethod


class BaseNeuroevolutionAlgorithm(object):
    """
    ToDo
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def create_initial_population(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement create_initial_population()")

    @abstractmethod
    def create_new_generation(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement create_new_generation()")

    @abstractmethod
    def check_population_extinction(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement check_population_extinction()")
