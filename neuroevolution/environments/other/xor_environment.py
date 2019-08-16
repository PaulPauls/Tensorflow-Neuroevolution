import ast
import numpy as np
import tensorflow as tf

from neuroevolution.environments import BaseEnvironment


class XOREnvironment(BaseEnvironment):

    def __init__(self, config):
        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

        self.input_shape = ast.literal_eval(config.get('ENVIRONMENT', 'input_shape', fallback='(2,)'))
        self.num_output = config.getint('ENVIRONMENT', 'num_output', fallback=1)

    def eval_genome_fitness(self, genome):
        # Get the phenotype model from the genome and declare the optimizer and loss_function
        model = genome.get_phenotype_model()
        optimizer = tf.keras.optimizers.SGD(lr=0.2)
        loss_function = tf.keras.losses.BinaryCrossentropy()

        # Compile and train the model
        model.compile(optimizer=optimizer, loss=loss_function)
        model.fit(self.x, self.y, batch_size=1, epochs=1000, verbose=0)

        # Calculate the fitness of the genome as the percentage of accuracy in its prediction
        evaluated_fitness = 1 - loss_function(self.y, model.predict(self.x))
        genome.set_fitness(evaluated_fitness)

    def replay_genome(self, genome):
        raise NotImplementedError("Should implement replay_genome()")

    def get_input_shape(self):
        return self.input_shape

    def get_num_output(self):
        return self.num_output
