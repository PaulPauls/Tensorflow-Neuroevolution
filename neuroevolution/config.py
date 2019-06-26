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

        self.parameters = self._load_parameters(config_path)
        self.logger.debug("Processed parameters in config ({}):\n{}".format(config_path, self.parameters))

    @staticmethod
    def _load_parameters(config_path):
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

        parameters_dict = {}
        for section in parameters.sections():
            parameters_dict[section] = dict(parameters.items(section))

        return parameters_dict
