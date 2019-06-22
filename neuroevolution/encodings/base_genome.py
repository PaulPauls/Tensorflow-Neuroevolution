"""
Base class for potential neuroevolution encoding genomes to subclass. This ensures that the encoding-genomes used in the
Tensorflow-Neuroevolution framework implement the required functions in the intended way.
"""

from abc import ABCMeta, abstractmethod


class BaseGenome(object):
    """
    ToDo
    """

    __metaclass__ = ABCMeta

    @abstractmethod
    def translate_to_phenotype(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("Should implement translate_to_phenotype()")
