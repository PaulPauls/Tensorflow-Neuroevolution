from abc import ABCMeta, abstractmethod


class BaseEncoding(object, metaclass=ABCMeta):
    @abstractmethod
    def create_new_genome(self, genotype, *args, **kwargs):
        raise NotImplementedError("Should implement create_new_genome()")
