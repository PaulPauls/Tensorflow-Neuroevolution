from abc import ABCMeta, abstractmethod

from tfne.encodings.base_genome import BaseGenome


class BaseNeuroevolutionAlgorithm(object, metaclass=ABCMeta):
    """
    Interface for TFNE compatible algorithms, which encapsulate the functionality of initialization, evaluation,
    evolution and serialization.
    """

    @abstractmethod
    def initialize_population(self, environment):
        """
        Initialize the population according to the specified NE algorithm. Adhere to potential constraints set by the
        environment.
        @param environment: one instance or multiple instances of the evaluation environment
        """
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement "
                                  "'initialize_population()'")

    @abstractmethod
    def evaluate_population(self, environment) -> (int, float):
        """
        Evaluate all members of the population on the supplied evaluation environment by passing each member to the
        environment and assigning the resulting fitness back to the member. Return the generation counter and the best
        achieved fitness.
        @param environment: one instance or multiple instances of the evaluation environment
        @return: tuple of generation counter and best fitness achieved by best member
        """
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'evaluate_population()'")

    @abstractmethod
    def summarize_population(self):
        """
        Print summary of the algorithm's population to stdout to inform about progress
        """
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'summarize_evaluation()'")

    @abstractmethod
    def evolve_population(self) -> bool:
        """
        Evolve all members of the population according to the NE algorithms specifications. Return a bool flag
        indicating if the population went extinct during the evolution
        @return: bool flag, indicating ig population went extinct during evolution
        """
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'evolve_population()'")

    @abstractmethod
    def save_state(self, save_dir_path):
        """
        Save the state of the algorithm and the current evolutionary process by serializing all aspects to json
        compatible dicts and saving it as file to the supplied save dir path.
        @param save_dir_path: string of directory path to which the state should be saved
        """
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'save_state()'")

    @abstractmethod
    def get_best_genome(self) -> BaseGenome:
        """
        @return: best genome so far determined by the evolutionary process
        """
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement 'get_best_genome()'")

    @abstractmethod
    def get_eval_instance_count(self) -> int:
        """
        @return: int, specifying how many evaluation threads the NE algorithm uses
        """
        raise NotImplementedError("Subclass of BaseNeuroevolutionAlgorithm does not implement "
                                  "'get_eval_instance_count()'")
