from abc import ABCMeta, abstractmethod


class BaseEncoding(object, metaclass=ABCMeta):
    """
    Base class for potential neuroevolution encodings to subclass. This ensures that the encodings used in the
    Tensorflow-Neuroevolution framework implement the required functions in the intended way.
    """

    @abstractmethod
    def create_new_genome(self, genotype, *args, **kwargs):
        raise NotImplementedError("Should implement create_new_genome()")
