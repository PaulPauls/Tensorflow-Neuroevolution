import gym

from neuroevolution.environments import BaseEnvironment


class CartPoleEnvironment(BaseEnvironment):
    """
    ToDo: Implement possibility that an algorithm will require multiple test environments either due to parallel
          execution (see batch_size) or the inability of the environment to be properly reset when testing a new genome.
          Therefore possibly put the creation of this class in the evolution_engine as it will know the batch_size.
    """
    def __init__(self):
        """
        ToDo
        """
        self.env = gym.make('CartPole-v1')

    def eval_genome_fitness(self, genome):
        """
        ToDo: Input genome; apply the genome to the test environments; Return its calculated resulting fitness value
        :param genome:
        :return:
        """
        pass

    def replay_genome(self, genome):
        """
        ToDo: Input genome, apply it to the test environment, though this time render the process of it being applied
        :param genome:
        :return: None
        """
        pass
