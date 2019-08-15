from neuroevolution.encodings import BaseEncoding
from neuroevolution.encodings.direct import DirectEncodingGenome


class DirectEncoding(BaseEncoding):

    def __init__(self, config):
        self.genome_id_counter = 0

        # Read in config parameters for the genome encoding
        self.check_genome_sanity = config.getboolean('ENCODING', 'check_genome_sanity')

    def create_new_genome(self, genotype):
        self.genome_id_counter += 1
        return DirectEncodingGenome(self.genome_id_counter, genotype, check_genome_sanity=self.check_genome_sanity)
