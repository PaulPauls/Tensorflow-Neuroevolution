import os
import tensorflow as tf
from configparser import ConfigParser


class Config:
    """
    ToDo
    """
    def __init__(self, config_path):
        """
        ToDo:
        :param config_path:
        """
        self.logger = tf.get_logger()

        self.algorithm_parameters = self._load_algorithm_parameters(config_path)
        self.logger.debug(self.algorithm_parameters)

    @staticmethod
    def _load_algorithm_parameters(config_path):
        """
        ToDo
        :param config_path:
        :return:
        """

        if not os.path.isfile(config_path):
            raise Exception("Specified configuration file does not exist: " + os.path.abspath(config_path))

        parameters = ConfigParser()
        with open(config_path) as config_file:
            parameters.read_file(config_file)

        algorithm_parameters = {}
        for section in parameters.sections():
            algorithm_parameters[section] = dict(parameters.items(section))

        return algorithm_parameters
