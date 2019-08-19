raise NotImplementedError
'''
Not yet compatible with Architecture Refactoring (TFNE Frameowrk still in early Alpha!)
'''

from neuroevolution.encodings import BaseEncoding
from neuroevolution.encodings.layer import KerasLayerEncodingGenome


class KerasLayerEncoding(BaseEncoding):
    def __init__(self, input_shape, num_output, config):
        self.input_shape = input_shape
        self.num_output = num_output

        self.genome_id_counter = 0

    def create_genome(self):
        genome = KerasLayerEncodingGenome(self.input_shape, self.num_output, self.genome_id_counter)
        self.genome_id_counter += 1
        return genome

    def pop_id(self):
        new_id = self.genome_id_counter
        self.genome_id_counter += 1
        return new_id
