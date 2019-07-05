import tensorflow as tf

from neuroevolution.encodings import BaseEncoding
from neuroevolution.encodings import BaseGenome


class DirectEncodedGenome(BaseGenome):

    def __init__(self, genome_id):
        pass


class DirectEncoding(BaseEncoding):

    def __init__(self, config):
        pass

    def create_genome(self):
        raise NotImplementedError("Should implement create_genome()")

    def pop_id(self):
        raise NotImplementedError("Should implement pop_id()")
