import ast
import numpy as np

from neuroevolution.environments import BaseEnvironment


class XOREnvironment(BaseEnvironment):

    def __init__(self, config):
        self.x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        self.y = np.array([[0], [1], [1], [0]])

        self.input_shape = ast.literal_eval(config.get('ENVIRONMENT', 'input_shape', fallback='(2,)'))
        self.num_output = config.getint('ENVIRONMENT', 'num_output', fallback=2)

    def eval_genome_fitness(self, genome):
        raise NotImplementedError("Should implement eval_genome_fitness()")

    def replay_genome(self, genome):
        raise NotImplementedError("Should implement replay_genome()")

    def get_input_shape(self):
        return self.input_shape

    def get_num_output(self):
        return self.num_output
