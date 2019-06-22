import tensorflow as tf


class Population:
    """
    ToDo
    """
    def __init__(self):
        """
        ToDo
        """
        self.logger = tf.get_logger()

        self.initialized_flag = False

    def get_best_genome(self):
        """
        ToDo
        :return:
        """
        pass

    def save_population(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("save_population() not yet implemented")

    def load_population(self):
        """
        ToDo
        :return:
        """
        raise NotImplementedError("load_population() not yet implemented")
