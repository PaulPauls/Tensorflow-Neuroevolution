from abc import ABCMeta, abstractmethod


class BasePopulation(object, metaclass=ABCMeta):
    """
    Interface for TFNE algorithm populations, which are supposed to hold all relevant population information in a single
    place to ease summary, serialization and deserialization.
    """

    @abstractmethod
    def summarize_population(self) -> dict:
        """
        Prints the current state of all population variables to stdout in a formatted and clear manner
        """
        raise NotImplementedError("Subclass of BasePopulation does not implement 'summarize_population()'")

    @abstractmethod
    def serialize(self) -> dict:
        """
        Serializes all population variables to a json compatible dictionary and returns it
        @return: serialized population variables as a json compatible dict
        """
        raise NotImplementedError("Subclass of BasePopulation does not implement 'serialize()'")
