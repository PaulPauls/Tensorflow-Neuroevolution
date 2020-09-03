from typing import Union, Any
from abc import ABCMeta, abstractmethod

import tensorflow as tf


class BaseGenome(object, metaclass=ABCMeta):
    """
    Interface for TFNE compatible genomes, which encapsulates all necessary functionality used by the algorithm,
    evaluation environment, visualizer, etc.
    """

    @abstractmethod
    def __call__(self, inputs) -> tf.Tensor:
        """
        Call genome to start inference based on the internal model. Return the results of the inference.
        @param inputs: genome model inputs
        @return: results of the genome model inference
        """
        raise NotImplementedError("Subclass of BaseGenome does not implement '__call__()'")

    @abstractmethod
    def __str__(self) -> str:
        """
        @return: string representation of the genome
        """
        raise NotImplementedError("Subclass of BaseGenome does not implement '__str__()'")

    @abstractmethod
    def visualize(self, show, save_dir_path, **kwargs) -> str:
        """
        Visualize the genome. If 'show' flag is set to true, display the genome after rendering. If 'save_dir_path' is
        supplied, save the rendered genome as file to that directory. Return the saved file path as string.
        @param show: bool flag, indicating whether the rendered genome should be displayed or not
        @param save_dir_path: string of the save directory path the rendered genome should be saved to.
        @param kwargs: Optional additional arguments relevant for rendering of the specific genome implementation.
        @return: string of the file path to which the rendered genome has been saved to
        """
        raise NotImplementedError("Subclass of BaseGenome does not implement 'visualize()'")

    @abstractmethod
    def serialize(self) -> dict:
        """
        @return: serialized constructor variables of the genome as json compatible dict
        """
        raise NotImplementedError("Subclass of BaseGenome does not implement 'serialize()'")

    @abstractmethod
    def save_genotype(self, save_dir_path) -> str:
        """
        Save genotype of genome to 'save_dir_path' directory. Return file path to which the genotype has been saved to
        as string.
        @param save_dir_path: string of the save directory path the genotype should be saved to
        @return: string of the file path to which the genotype has been saved to
        """
        raise NotImplementedError("Subclass of BaseGenome does not implement 'save_genotype()'")

    @abstractmethod
    def save_model(self, file_path, **kwargs):
        """
        Save TF model of genome to specified file path.
        @param file_path: string of the file path the TF model should be saved to
        @param kwargs: Optional additional arguments relevant for TF model.save()
        """
        raise NotImplementedError("Subclass of BaseGenome does not implement 'save_model()'")

    @abstractmethod
    def set_fitness(self, fitness):
        """
        Set genome fitness value to supplied parameter
        @param fitness: float of genome fitness
        """
        raise NotImplementedError("Subclass of BaseGenome does not implement 'set_fitness()'")

    @abstractmethod
    def get_genotype(self) -> Any:
        """
        @return: One or multiple variables representing the genome genotype
        """
        raise NotImplementedError("Subclass of BaseGenome does not implement 'get_genotype()'")

    @abstractmethod
    def get_model(self) -> tf.keras.Model:
        """
        @return: TF model represented by genome genotype
        """
        raise NotImplementedError("Subclass of BaseGenome does not implement 'get_model()'")

    @abstractmethod
    def get_optimizer(self) -> Union[None, tf.keras.optimizers.Optimizer]:
        """
        Return either None or TF optimizer depending on if the genome encoding associates an optimizer with the genome
        @return: None | TF optimizer associated with genome
        """
        raise NotImplementedError("Subclass of BaseGenome does not implement 'get_optimizer()'")

    @abstractmethod
    def get_id(self) -> int:
        """
        @return: int of genome ID
        """
        raise NotImplementedError("Subclass of BaseGenome does not implement 'get_id()'")

    @abstractmethod
    def get_fitness(self) -> float:
        """
        @return: float of genome fitness
        """
        raise NotImplementedError("Subclass of BaseGenome does not implement 'get_fitness()'")
