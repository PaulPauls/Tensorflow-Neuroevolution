from abc import ABCMeta, abstractmethod


class BaseEncoding(object, metaclass=ABCMeta):

    @abstractmethod
    def create_genome(self):
        raise NotImplementedError("Should implement create_genome()")

    @abstractmethod
    def pop_id(self):
        raise NotImplementedError("Should implement pop_id()")
