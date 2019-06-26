import tensorflow as tf

from neuroevolution.environments import BaseEnvironment


class FashionMNISTEnvironment(BaseEnvironment):
    """
    ToDo: Implement possibility that an algorithm will require multiple test environments either due to parallel
          execution (see batch_size) or the inability of the environment to be properly reset when testing a new genome.
          Therefore possibly put the creation of this class in the evolution_engine as it will know the batch_size.
    """
    def __init__(self, config):
        """
        ToDo
        :param config:
        """
        self.logger = tf.get_logger()

        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, self.train_labels), (test_images, self.test_labels) = fashion_mnist.load_data()
        self.train_images = train_images / 255.0
        self.test_images = test_images / 255.0

        # Read in config parameters for environment
        self.train_epochs = int(config.get('FashionMNISTEnvironment', 'train_epochs'))

    def eval_genome_fitness(self, genome):
        """
        ToDo: Input genome; apply the genome to the test environments; Return its calculated resulting fitness value
        :param genome:
        :return:
        """
        model = genome.to_phenotype()
        model.fit(self.train_images, self.train_labels, epochs=self.train_epochs, verbose=0)
        _, test_accuracy = model.evaluate(self.test_images, self.test_labels)
        self.logger.debug("Genome {} scored test_accuracy: {}".format(genome.get_id(), test_accuracy))
        return test_accuracy

    def replay_genome(self, genome):
        """
        ToDo: Input genome, apply it to the test environment, though this time render the process of it being applied
        :param genome:
        :return: None
        """
        model = genome.to_phenotype()
        _, test_accuracy = model.evaluate(self.test_images, self.test_labels)
        self.logger.debug("Best Genome (Nr. {}) scored test_accuracy: {}".format(genome.get_id(), test_accuracy))
        model.summary()

    def get_input_shape(self):
        """
        ToDo
        :return:
        """
        return 28, 28

    def get_num_output(self):
        """
        ToDo
        :return:
        """
        return 10
