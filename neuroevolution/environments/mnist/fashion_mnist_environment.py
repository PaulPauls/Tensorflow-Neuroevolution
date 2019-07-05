import tensorflow as tf

from neuroevolution.environments import BaseEnvironment


class FashionMNISTEnvironment(BaseEnvironment):

    def __init__(self, config):
        self.logger = tf.get_logger()

        fashion_mnist = tf.keras.datasets.fashion_mnist
        (train_images, self.train_labels), (test_images, self.test_labels) = fashion_mnist.load_data()
        self.train_images = train_images / 255.0
        self.test_images = test_images / 255.0

        # Read in config parameters for environment
        self.train_epochs = int(config.get('FashionMNISTEnvironment', 'train_epochs'))

    def eval_genome_fitness(self, genome):
        model = genome.to_phenotype()
        model.fit(self.train_images, self.train_labels, epochs=self.train_epochs, verbose=0)
        _, test_accuracy = model.evaluate(self.test_images, self.test_labels)
        self.logger.debug("Genome {} scored test_accuracy: {}".format(genome.get_id(), test_accuracy))
        return test_accuracy

    def replay_genome(self, genome):
        model = genome.to_phenotype()
        _, test_accuracy = model.evaluate(self.test_images, self.test_labels)
        self.logger.debug("Best Genome (Nr. {}) scored test_accuracy: {}".format(genome.get_id(), test_accuracy))
        model.summary()

    def get_input_shape(self):
        return 28, 28

    def get_num_output(self):
        return 10
