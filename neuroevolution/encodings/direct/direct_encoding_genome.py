from neuroevolution.encodings import BaseGenome
from neuroevolution.encodings.direct import DirectEncodingModel, visualize_direct_encoding_genome


class NEATLikeGene:
    """ Possibly also create a BaseGene class and make this subclass of it. Depending on if translation function
        needs a fixed Gene structure """
    def __init__(self, gene_id, conn_in, conn_out, activation=None, next_gene=None):
        self.gene_id = gene_id
        self.conn_in = conn_in
        self.conn_out = conn_out
        self.activation = activation
        self.next_gene = next_gene


class DirectEncodingGenome(BaseGenome):
    def __init__(self, genome_id, genotype, inputs_outpus=None, activations=None, check_genome_sanity=False):
        self.genome_id = genome_id
        self.fitness = 0
        if isinstance(genotype, NEATLikeGene):
            self.genotype = genotype
            self.inputs_outputs = inputs_outpus
            self.activations = activations
        else:
            self.genotype, self.inputs_outputs, self.activations = \
                self._deserialize_genotype(genotype, inputs_outpus, activations)
        if check_genome_sanity:
            self._check_genome_sanity()
        self.phenotype_model = self._create_phenotype_model(trainable=True)

    def __call__(self, *args, **kwargs):
        return self.phenotype_model(*args, **kwargs)

    def __str__(self):
        string_repr = "Genome-ID: {}, Fitness: {}, Genotype: {}".format(self.genome_id, self.fitness,
                                                                        self.serialize_genotype())
        return string_repr

    def serialize_genotype(self):
        serialzed_genome = dict()
        if self.inputs_outputs or self.activations:
            serialzed_genome[0] = {**self.inputs_outputs, **self.activations}

        gene = self.genotype
        while gene:
            if gene.activation:
                serialized_gene = (gene.conn_in, gene.conn_out, gene.activation)
            else:
                serialized_gene = (gene.conn_in, gene.conn_out)
            serialzed_genome[gene.gene_id] = serialized_gene
            gene = gene.next_gene

        return serialzed_genome

    def summary(self):
        print(self)
        # If phenotype_model build, do: print(self.phenotype_model.summary())

    def visualize(self):
        return visualize_direct_encoding_genome(self)

    def get_phenotype_model(self):
        return self.phenotype_model

    def get_id(self):
        return self.genome_id

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_fitness(self):
        return self.fitness

    def _check_genome_sanity(self):
        """
        TODO: Extend the Genotype checks performed in this function. Checks to Implement:
        - if activations are specified in genes, check that there are no different activations for same output node

        As of now checking for:
        - gene_id's unique
        - inputs_outputs and activations have each required field
        - inputs_outputs specified and connections actually present in genes
        - either 'activations' specified and no gene has specific activation or each gene has specific activations and
          no 'activations' specified
        """
        gene = self.genotype
        unique_ids = set()
        while gene:
            if gene.gene_id in unique_ids:
                raise Exception("gene_id {} present multiple times in single genome".format(gene.gene_id))
            unique_ids.add(gene.gene_id)
            gene = gene.next_gene

        assert self.inputs_outputs['inputs'] is not None
        assert self.inputs_outputs['outputs'] is not None
        assert self.activations['layer_activations'] is not None
        assert self.activations['out_activation'] is not None
        assert self.activations['default_activation'] is not None

        gene = self.genotype
        inputs = set(self.inputs_outputs['inputs'])
        outputs = set(self.inputs_outputs['outputs'])
        while gene:
            inputs.discard(gene.conn_in)
            outputs.discard(gene.conn_out)
            gene = gene.next_gene
        if (len(inputs) > 0) or (len(outputs) > 0):
            raise Exception("input or output connection specified in 'input_outputs' not present in genotype")

        if self.activations:
            gene = self.genotype
            while gene:
                if gene.activation:
                    raise Exception("Gene with ID {} has activation even though global 'activations' specified".
                                    format(gene.gene_id))
                gene = gene.next_gene
        else:
            gene = self.genotype
            while gene:
                if not gene.activation:
                    raise Exception("Gene with ID {} has no activation even though no global 'activations' specified".
                                    format(gene.gene_id))
                gene = gene.next_gene

    def _create_phenotype_model(self, trainable):
        return DirectEncodingModel(self.genotype, self.inputs_outputs, self.activations, trainable=trainable)

    @staticmethod
    def _deserialize_genotype(genotype, inputs_outputs, activations):
        exception_msg = None
        try:
            inputs = genotype[0]['inputs']
            outputs = genotype[0]['outputs']
            layer_activations = genotype[0]['layer_activations']
            out_activation = genotype[0]['out_activation']
            default_activation = genotype[0]['default_activation']

            if inputs_outputs or activations:
                exception_msg = "'inputs_outputs' and 'activations' defined double in genotype and passed parameters."
                raise ValueError()
            inputs_outputs = {'inputs': inputs, 'outputs': outputs}
            activations = {'layer_activations': layer_activations, 'out_activation': out_activation,
                           'default_activation': default_activation}
        except (KeyError, ValueError):
            # TODO: Exception is not raised if 'inputs_outputs' or 'activations' are only defined partially
            if not inputs_outputs or not activations:
                exception_msg = "genotype and passed parameters don't define either 'input_outputs' or 'activations'"
            if exception_msg:
                raise Exception(exception_msg)
        del genotype[0]

        preceding_gene = None
        head_gene = None
        for gene_id, conns in genotype.items():
            if len(conns) == 2:
                new_gene = NEATLikeGene(gene_id, conns[0], conns[1])
            else:
                new_gene = NEATLikeGene(gene_id, conns[0], conns[1], activation=conns[2])
            if preceding_gene:
                preceding_gene.next_gene = new_gene
            else:
                head_gene = new_gene
            preceding_gene = new_gene

        return head_gene, inputs_outputs, activations
