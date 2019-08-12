from neuroevolution.environments import BaseEnvironment


class XOREnvironment(BaseEnvironment):

    def __init__(self, config):
        raise NotImplementedError("Should implement init()")

    def eval_genome_fitness(self, genome):
        raise NotImplementedError("Should implement eval_genome_fitness()")

    def replay_genome(self, genome):
        raise NotImplementedError("Should implement replay_genome()")

    def get_input_shape(self):
        raise NotImplementedError("Should implement get_input_shape()")

    def get_num_output(self):
        raise NotImplementedError("Should implement get_num_output()")
