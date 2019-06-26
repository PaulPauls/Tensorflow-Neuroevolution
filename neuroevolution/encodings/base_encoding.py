from abc import ABCMeta, abstractmethod


class BaseEncoding(object):
    """
    ToDo
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def create_genome(self, genome_id):
        """
        ToDo
        :param genome_id:
        :return:
        """
        raise NotImplementedError("Should implement create_genome()")
