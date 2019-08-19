import gym

from neuroevolution.environments import BaseEnvironment


class CartPoleEnvironment(BaseEnvironment):

    def __init__(self, config):
        self.env = gym.make("CartPole-v1")

    def eval_genome_fitness(self, genome):
        raise NotImplementedError("Should implement eval_genome_fitness()")

    def replay_genome(self, genome):
        raise NotImplementedError("Should implement replay_genome()")

    def get_input_shape(self):
        return self.env.observation_space.shape

    def get_num_output(self):
        return self.env.action_space.n
