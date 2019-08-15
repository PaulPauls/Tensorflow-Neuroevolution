from neuroevolution.encodings import BaseEncoding
from neuroevolution.encodings.direct import DirectEncodingGenome


def serialize_genome(genotype, activations):
    pass
    #raise NotImplementedError()

def deserialize_genome(genotype, activations):
    pass
    #raise NotImplementedError()

def check_genome_sanity_function(genotype, activations):
    pass
    #raise NotImplementedError()

class DirectEncoding(BaseEncoding):

    def __init__(self, config):
        self.genome_id_counter = 0

        # Read in config parameters for the genome encoding
        self.check_genome_sanity = config.getboolean('ENCODING', 'check_genome_sanity')

    def create_new_genome(self, genotype, activations, trainable=True, check_genome_sanity=None):
        check_genome_sanity = self.check_genome_sanity if check_genome_sanity is None else check_genome_sanity

        # if genotype is explicitely specified in dict form it will be deserialized into a linked-List. Otherwise
        # we assume that the genotype is in the correct form (deque of DirectEncodingGenes). If this is not the
        # case will check_genome_sanity() throw an error
        if isinstance(genotype, dict):
            genotype = deserialize_genome(genotype, activations)

        if check_genome_sanity:
            check_genome_sanity_function(genotype, activations)

        self.genome_id_counter += 1
        return DirectEncodingGenome(self.genome_id_counter, genotype, activations, trainable=trainable)
