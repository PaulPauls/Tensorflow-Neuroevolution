from abc import ABCMeta, abstractmethod


class BaseEncoding(object):
    """
    ToDo
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def create_genome(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement create_genome()")

    @abstractmethod
    def pop_id(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement pop_id()")
