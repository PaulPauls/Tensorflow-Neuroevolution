from neuroevolution.algorithms.base_algorithm import BaseNeuroevolutionAlgorithm


class Config:
    """
    ToDo
    """
    def __init__(self, ne_algorithm, config_path):
        """
        ToDo: Create class of specified ne_algorithm; Check if the ne_algorithm is subclass of 'abstract_algorithm' and
              therefore implements all required methods; Load and save configuration in thsi config
        :param ne_algorithm:
        :param config_path:
        """
        assert issubclass(ne_algorithm, BaseNeuroevolutionAlgorithm)
        self.ne_algorithm = ne_algorithm()

        self.algorithm_parameters = self._load_algorithm_parameters()
        self.genome_parameters = self._load_genome_parameters()
        self.reproduction_parameters = self._load_reproduction_parameters()
        self.speciation_parameters = self._load_speciation_parameters()
        self.stagnation_parameters = self._load_stagnation_parameters()

        pass

    def _load_algorithm_parameters(self):
        """
        ToDo: Load Hyperparameters of the algorithm itself. E.g., the section in the config that is named after the
              algorithm. For the NEAT algorithm would this function therefore load the [NEAT] section.
        ToDo
        :return:
        """
        pass

    def _load_genome_parameters(self):
        """
        ToDo
        :return:
        """
        pass

    def _load_reproduction_parameters(self):
        """
        ToDo
        :return:
        """
        pass

    def _load_speciation_parameters(self):
        """
        ToDo
        :return:
        """
        pass

    def _load_stagnation_parameters(self):
        """
        ToDo
        :return:
        """
        pass
