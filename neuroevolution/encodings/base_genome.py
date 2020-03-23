import tensorflow as tf
from abc import ABCMeta, abstractmethod


class BaseGenome(object, metaclass=ABCMeta):

    @abstractmethod
    def __str__(self) -> str:
        raise NotImplementedError("Should implement __str__()")

    @abstractmethod
    def visualize(self, view, render_file_path):
        """
        Display rendered genome or save rendered genome to specified path or do both
        :param view: flag if rendered genome should be displayed
        :param render_file_path: string of file path, specifying where the genome render should be saved
        """
        raise NotImplementedError("Should implement visualize()")

    @abstractmethod
    def get_model(self) -> tf.keras.Model:
        """
        :return: Tensorflow model phenotype translation of the genome genotype
        """
        raise NotImplementedError("Should implement visualize()")

    @abstractmethod
    def get_genotype(self) -> dict:
        """
        :return: genome genotype dict with the keys being the gene-ids and the values being the genes
        """
        raise NotImplementedError("Should implement visualize()")

    @abstractmethod
    def get_id(self) -> int:
        raise NotImplementedError("Should implement visualize()")

    @abstractmethod
    def get_fitness(self) -> float:
        raise NotImplementedError("Should implement visualize()")

    @abstractmethod
    def set_fitness(self, fitness):
        raise NotImplementedError("Should implement visualize()")
