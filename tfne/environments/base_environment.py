from __future__ import annotations

from abc import ABCMeta, abstractmethod


class BaseEnvironment(object, metaclass=ABCMeta):
    """
    Interface for TFNE compatible environments, which are supposed to encapsulate a problem and provide the necessary
    information and functions that the TFNE pre-implemented algorithms require.
    """

    @abstractmethod
    def eval_genome_fitness(self, genome) -> float:
        """
        Evaluates the genome's fitness in either the weight-training or non-weight-training variant. Returns the
        determined genome fitness.
        @param genome: TFNE compatible genome that is to be evaluated
        @return: genome calculated fitness
        """
        raise NotImplementedError("Subclass of BaseEnvironment does not implement 'eval_genome_fitness()'")

    @abstractmethod
    def replay_genome(self, genome):
        """
        Replay genome on environment by calculating its fitness and printing it.
        @param genome: TFNE compatible genome that is to be evaluated
        """
        raise NotImplementedError("Subclass of BaseEnvironment does not implement 'replay_genome()'")

    @abstractmethod
    def duplicate(self) -> BaseEnvironment:
        """
        @return: New instance of the environment with identical parameters
        """
        raise NotImplementedError("Subclass of BaseEnvironment does not implement 'duplicate()'")

    @abstractmethod
    def get_input_shape(self) -> (int, ...):
        """
        @return: Environment input shape that is required from the applied TF models
        """
        raise NotImplementedError("Subclass of BaseEnvironment does not implement 'get_input_shape()'")

    @abstractmethod
    def get_output_shape(self) -> (int, ...):
        """
        @return: Environment output shape that is required from the applied TF models
        """
        raise NotImplementedError("Subclass of BaseEnvironment does not implement 'get_output_shape()'")
