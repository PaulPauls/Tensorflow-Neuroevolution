from abc import ABCMeta, abstractmethod


class BaseEnvironment(object):
    """
    ToDo
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def eval_genome_fitness(self, genome):
        """
        ToDo: Input genome; apply the genome to the test environments; Return its calculated resulting fitness value
        :param genome:
        :return:
        """
        raise NotImplementedError("Should implement eval_genome_fitness()")

    @abstractmethod
    def replay_genome(self, genome):
        """
        ToDo: Input genome, apply it to the test environment, though this time render the process of it being applied
        :param genome:
        :return: None
        """
        raise NotImplementedError("Should implement replay_genome()")

    @abstractmethod
    def get_input_shape(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement get_input_shape()")

    @abstractmethod
    def get_num_output(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement get_num_output()")
