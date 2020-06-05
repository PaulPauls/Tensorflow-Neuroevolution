from abc import ABCMeta, abstractmethod

from .base_genome import BaseGenome


class BaseEncoding(object, metaclass=ABCMeta):
    """"""

    @abstractmethod
    def create_genome(self, *args) -> (int, BaseGenome):
        """"""
        raise NotImplementedError("Subclass of BaseEncoding does not implement 'create_genome()'")
