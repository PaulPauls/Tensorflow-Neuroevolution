from abc import ABCMeta, abstractmethod


class BaseEnvironment(object, metaclass=ABCMeta):

    @abstractmethod
    def eval_genome_fitness(self, genome) -> float:
        """
        Calculate and return the genome fitness as the accuracy in its ability to predict the correct solution for the
        environment.
        :param genome: genome of the TFNE framework, providing a built Tensorflow model
        :return: genome model accuracy of predicting the correct solution for the environment
        """
        raise NotImplementedError("Should implement eval_genome_fitness()")

    @abstractmethod
    def replay_genome(self, genome):
        """
        Replay the genome by demonstrating its ability to solve the environment.
        :param genome: genome of the TFNE framework, providing a built Tensorflow model
        """
        raise NotImplementedError("Should implement replay_genome()")

    @abstractmethod
    def get_input_shape(self) -> ():
        """
        :return: multi-dimensional tuple specifying the shape of the inputs for a model supplied to this environment
        """
        raise NotImplementedError("Should implement get_input_shape()")

    @abstractmethod
    def get_num_output(self) -> int:
        """
        :return: number of features of an environment the genome model should predict
        """
        raise NotImplementedError("Should implement get_num_output()")
