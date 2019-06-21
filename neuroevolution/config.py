import tensorflow as tf


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

        self.algorithm_parameters = self._load_algorithm_parameters()

        pass

    def _load_algorithm_parameters(self):
        """
        ToDo
        :return:
        """
        pass
