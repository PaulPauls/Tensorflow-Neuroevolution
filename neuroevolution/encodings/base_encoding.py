"""
Base class for potential neuroevolution encoding genomes to subclass. This ensures that the encoding-genomes used in the
Tensorflow-Neuroevolution framework implement the required functions in the intended way.
"""

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
