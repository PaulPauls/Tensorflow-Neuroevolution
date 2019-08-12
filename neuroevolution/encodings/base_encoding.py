"""
Base class for potential neuroevolution encodings to subclass. This ensures that the encodings used in the
Tensorflow-Neuroevolution framework implement the required functions in the intended way.
"""

from abc import ABCMeta, abstractmethod


class BaseEncoding(object, metaclass=ABCMeta):
    pass
