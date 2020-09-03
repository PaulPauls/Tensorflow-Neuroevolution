from __future__ import annotations

from abc import ABCMeta, abstractmethod
import tensorflow as tf


class CoDeepNEATModuleBase(object, metaclass=ABCMeta):
    """
    Base class and interface for TFNE CoDeepNEAT compatible modules, ensuring that modules provide layer creation,
    downsampling, mutation and crossover functionality. This base class also provides common functionality required
    by all modules like parameter saving and simple setter/getter methods.
    """

    def __init__(self, config_params, module_id, parent_mutation, dtype):
        """
        Base class of all TFNE CoDeepNEAT modules, saving common parameters.
        @param config_params: dict of the module parameter range supplied via config
        @param module_id: int of unique module ID
        @param parent_mutation: dict summarizing the mutation of the parent module
        @param dtype: string of deserializable TF dtype
        """
        self.config_params = config_params
        self.module_id = module_id
        self.parent_mutation = parent_mutation
        self.dtype = dtype
        self.fitness = 0

    @abstractmethod
    def __str__(self) -> str:
        """
        @return: string representation of the module
        """
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement '__str__()'")

    @abstractmethod
    def create_module_layers(self) -> (tf.keras.layers.Layer, ...):
        """
        Instantiate all TF layers represented by the module and return as iterable tuple
        @return: iterable tuple of instantiated TF layers
        """
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'create_module_layers()'")

    @abstractmethod
    def create_downsampling_layer(self, in_shape, out_shape) -> tf.keras.layers.Layer:
        """
        Create layer associated with this module that downsamples the non compatible input shape to the input shape of
        the current module, which is the output shape of the downsampling layer.
        @param in_shape: int tuple of incompatible input shape
        @param out_shape: int tuple of the intended output shape of the downsampling layer
        @return: instantiated TF keras layer that can downsample incompatible input shape to a compatible input shape
        """
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'create_downsampling_layer()'")

    @abstractmethod
    def create_mutation(self,
                        offspring_id,
                        max_degree_of_mutation) -> CoDeepNEATModuleBase:
        """
        Create a mutated module and return it
        @param offspring_id: int of unique module ID of the offspring
        @param max_degree_of_mutation: float between 0 and 1 specifying the maximum degree of mutation
        @return: instantiated TFNE CoDeepNEAT module with mutated parameters
        """
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'create_mutation()'")

    @abstractmethod
    def create_crossover(self,
                         offspring_id,
                         less_fit_module,
                         max_degree_of_mutation) -> CoDeepNEATModuleBase:
        """
        Create a crossed over module and return it
        @param offspring_id: int of unique module ID of the offspring
        @param less_fit_module: second module of same type with less fitness
        @param max_degree_of_mutation: float between 0 and 1 specifying the maximum degree of mutation
        @return: instantiated TFNE CoDeepNEAT module with crossed over parameters
        """
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'create_crossover()'")

    @abstractmethod
    def serialize(self) -> dict:
        """
        @return: serialized constructor variables of the module as json compatible dict
        """
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'serialize()'")

    @abstractmethod
    def get_distance(self, other_module) -> float:
        """
        Calculate the distance between 2 TFNE CoDeepNEAT modules with high values indicating difference, low values
        indicating similarity
        @param other_module: second TFNE CoDeepNEAT module to which the distance has to be calculated
        @return: float between 0 and 1. High values indicating difference, low values indicating similarity
        """
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'get_distance()'")

    @abstractmethod
    def get_module_type(self) -> str:
        """
        @return: string representation of module type as used in CoDeepNEAT config
        """
        raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'get_module_name()'")

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_id(self) -> int:
        return self.module_id

    def get_fitness(self) -> float:
        return self.fitness

    def get_merge_method(self) -> dict:
        return self.merge_method
