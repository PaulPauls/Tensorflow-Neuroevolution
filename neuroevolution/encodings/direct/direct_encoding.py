from neuroevolution.encodings import BaseEncoding
from neuroevolution.encodings.direct import DirectEncodingGenome


class DirectEncoding(BaseEncoding):

    def __init__(self, config):
        self.genome_id_counter = 0

    def create_new_genome(self, genotype):
        self.genome_id_counter += 1
        return DirectEncodingGenome(self.genome_id_counter, genotype)
