import numpy as np
import tensorflow as tf
from absl import logging
from copy import deepcopy
from random import choice, random, randint, shuffle

import neuroevolution as ne
from ..base_algorithm import BaseNeuroevolutionAlgorithm
from ...encodings.direct.direct_encoding_genome import DirectEncodingGenome


class NEAT(BaseNeuroevolutionAlgorithm):
    """
    Implementation of Kenneth O'Stanleys and Risto Miikkulainen's algorithm 'Neuroevolution of Augmenting Topologies'
    (NEAT) [1,2] for the Tensorflow-Neuroevolution framework.
    [1] http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
    [2] http://nn.cs.utexas.edu/downloads/papers/stanley.phd04.pdf
    """

    def __init__(self, config, dtype=tf.float32, run_eagerly=False):
        # Declare, read in and log config parameters for the NEAT algorithm
        self.reproducing_fraction = None
        self.crossover_prob = None
        self.mutation_weights_prob = None
        self.mutation_add_conn_prob = None
        self.mutation_add_node_prob = None
        self.mutation_weights_fraction = None
        self.mutation_weights_mean = None
        self.mutation_weights_stddev = None
        self.distance_excess_c1 = None
        self.distance_disjoint_c2 = None
        self.distance_weight_c3 = None
        self.activation_hidden = None
        self.activation_output = None
        self.species_elitism = None
        self.species_max_stagnation = None
        self.species_clustering = None
        self._read_config_parameters(config)
        self._log_class_parameters()

        # Assert that the probabilities of crossover or mutation add to 100%, as either one is always chosen
        assert self.crossover_prob + self.mutation_weights_prob + \
               self.mutation_add_conn_prob + self.mutation_add_node_prob == 1.0

        # Use DirectEncoding with the supplied parameters for NEAT. Set trainable to False as NEAT is training/evolving
        # the weights itself
        self.encoding = ne.encodings.DirectEncoding(trainable=False, dtype=dtype, run_eagerly=run_eagerly)

        # Initialize species containers. Start with species_id_counter set to 1 as the population initialization will
        # assign all newly initialized genomes to species 1, as defined per NEAT.
        self.species_id_counter = 1
        self.species_assignment = dict()
        self.species_avg_fitness_history = dict()

        # Initialize implementation specific dicts, keeping track of added nodes and the adjusted fitness of genomes,
        # required for determining alloted offspring of each species.
        self.node_counter = None
        self.add_node_history = dict()
        self.genomes_adj_fitness = dict()

    def _read_config_parameters(self, config):
        """
        Read the class parameters supplied via the config file
        :param config: ConfigParser Object which has processed the supplied configuration
        """
        section_name_algorithm = 'NEAT' if config.has_section('NEAT') else 'NE_ALGORITHM'
        self.reproducing_fraction = config.getfloat(section_name_algorithm, 'reproducing_fraction')
        self.crossover_prob = config.getfloat(section_name_algorithm, 'crossover_prob')
        self.mutation_weights_prob = config.getfloat(section_name_algorithm, 'mutation_weights_prob')
        self.mutation_add_conn_prob = config.getfloat(section_name_algorithm, 'mutation_add_conn_prob')
        self.mutation_add_node_prob = config.getfloat(section_name_algorithm, 'mutation_add_node_prob')
        self.mutation_weights_fraction = config.getfloat(section_name_algorithm, 'mutation_weights_fraction')
        self.mutation_weights_mean = config.getfloat(section_name_algorithm, 'mutation_weights_mean')
        self.mutation_weights_stddev = config.getfloat(section_name_algorithm, 'mutation_weights_stddev')
        self.distance_excess_c1 = config.getfloat(section_name_algorithm, 'distance_excess_c1')
        self.distance_disjoint_c2 = config.getfloat(section_name_algorithm, 'distance_disjoint_c2')
        self.distance_weight_c3 = config.getfloat(section_name_algorithm, 'distance_weight_c3')
        self.species_elitism = config.getint(section_name_algorithm, 'species_elitism')
        self.species_max_stagnation = config.get(section_name_algorithm, 'species_max_stagnation')
        self.species_clustering = config.get(section_name_algorithm, 'species_clustering')
        self.activation_hidden = config.get(section_name_algorithm, 'activation_hidden')
        self.activation_output = config.get(section_name_algorithm, 'activation_output')

        if ',' in self.species_clustering:
            species_clustering_alg = self.species_clustering[:self.species_clustering.find(',')]
            species_clustering_val = float(self.species_clustering[self.species_clustering.find(',') + 2:])
            self.species_clustering = (species_clustering_alg, species_clustering_val)

        if ',' in self.species_max_stagnation:
            species_max_stagnation_duration = int(self.species_max_stagnation[:self.species_max_stagnation.find(',')])
            species_max_stagnation_perc = float(self.species_max_stagnation[self.species_max_stagnation.find(',') + 2:])
            self.species_max_stagnation = (species_max_stagnation_duration, species_max_stagnation_perc)

        self.activation_hidden = tf.keras.activations.deserialize(self.activation_hidden)
        self.activation_output = tf.keras.activations.deserialize(self.activation_output)

    def _log_class_parameters(self):
        logging.debug("NEAT algorithm config: reproducing_fraction = {}".format(self.reproducing_fraction))
        logging.debug("NEAT algorithm config: crossover_prob = {}".format(self.crossover_prob))
        logging.debug("NEAT algorithm config: mutation_weights_prob = {}".format(self.mutation_weights_prob))
        logging.debug("NEAT algorithm config: mutation_add_conn_prob = {}".format(self.mutation_add_conn_prob))
        logging.debug("NEAT algorithm config: mutation_add_node_prob = {}".format(self.mutation_add_node_prob))
        logging.debug("NEAT algorithm config: mutation_weights_fraction = {}".format(self.mutation_weights_fraction))
        logging.debug("NEAT algorithm config: mutation_weights_mean = {}".format(self.mutation_weights_mean))
        logging.debug("NEAT algorithm config: mutation_weights_stddev = {}".format(self.mutation_weights_stddev))
        logging.debug("NEAT algorithm config: distance_excess_c1 = {}".format(self.distance_excess_c1))
        logging.debug("NEAT algorithm config: distance_disjoint_c2 = {}".format(self.distance_disjoint_c2))
        logging.debug("NEAT algorithm config: distance_weight_c3 = {}".format(self.distance_weight_c3))
        logging.debug("NEAT algorithm config: species_elitism = {}".format(self.species_elitism))
        logging.debug("NEAT algorithm config: species_max_stagnation = {}".format(self.species_max_stagnation))
        logging.debug("NEAT algorithm config: species_clustering = {}".format(self.species_clustering))
        logging.debug("NEAT algorithm config: activation_hidden = {}".format(self.activation_hidden))
        logging.debug("NEAT algorithm config: activation_output = {}".format(self.activation_output))

    def initialize_population(self, population, initial_pop_size, input_shape, num_output):
        """
        Initialize the population with DirectEncoding genomes created according to the NEATs specification of minimal,
        fully-connected topologies (not hidden nodes, all inputs are connected to all outputs). The initial connection
        weights are randomized in the sense of them being mutated once (which means the addition of a value from a
        random normal distribution with cfg specified mean and stddev), while the node biases are all initialized to 0.
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        :param initial_pop_size: int, amount of genomes to be initialized and added to the population
        :param input_shape: tuple, shape of the input vector for the NN model to be created
        :param num_output: int, number of nodes in the output layer of the NN model to be created
        """
        if len(input_shape) == 1:
            num_input = input_shape[0]
            # Create species 1, as population initialization assigns first created genome to this standard species
            self.species_assignment[1] = None
            self.species_avg_fitness_history[1] = []

            for _ in range(initial_pop_size):
                genotype = dict()
                for conn_in in range(1, num_input + 1):
                    for conn_out in range(num_input + 1, num_input + num_output + 1):
                        # Create initial connection weight as random value from normal distribution with mean and stddev
                        # as configured in the cfg, effectively setting the weight to 0 and mutating it once.
                        conn_weight = np.random.normal(loc=self.mutation_weights_mean,
                                                       scale=self.mutation_weights_stddev)
                        gene_id, gene_conn = self.encoding.create_gene_connection(conn_in, conn_out, conn_weight)
                        genotype[gene_id] = gene_conn
                for node in range(num_input + 1, num_input + num_output + 1):
                    # As each node created in this initialization is a node of the output layer, assign the output
                    # activation to all nodes.
                    gene_id, gene_node = self.encoding.create_gene_node(node, 0, self.activation_output)
                    genotype[gene_id] = gene_node

                # Set node counter to initialized nodes
                self.node_counter = num_input + num_output

                new_genome_id, new_genome = self.encoding.create_genome(genotype)
                population.add_genome(new_genome_id, new_genome)
                if self.species_assignment[1] is None:
                    self.species_assignment[1] = [new_genome_id]

        else:
            raise NotImplementedError("Multidimensional input vectors not yet supported")

    def evolve_population(self, population, pop_size_fixed):
        """
        Evolve the population by first removing stagnating species and then creating mutations (crossover, weight-
        mutation, add conn mutation, add node mutation) within the existing species, which in turn completely replace
        the old generation (except for the species champions).
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        :param pop_size_fixed: bool flag, indicating if the size of the population can be different after the evolution
                               of the current generation is complete
        """
        # Assert that pop size is fixed as NEAT does not operate on dynamic pop sizes
        assert pop_size_fixed

        # Determine order in which to go through species and evaluate if species should be removed or not. Evaluate
        # this stagnation from the least to the most fit species, judged on its avg fitness.
        sorted_species_ids = sorted(self.species_assignment.keys(),
                                    key=lambda x: self.species_avg_fitness_history[x][-1])
        max_stag_dur = self.species_max_stagnation[0]
        non_stag_improv = self.species_max_stagnation[1]

        # Remove stagnating species
        for species_id in sorted_species_ids:
            # Break stagnation evaluation if less species present than configured in 'species_elitism' cfg parameter
            if len(self.species_assignment) <= self.species_elitism:
                break
            # Only consider stagnation evaluation if species existed for at least 'max_stag_dur' generations
            avg_fitness_history = self.species_avg_fitness_history[species_id]
            if len(avg_fitness_history) >= max_stag_dur:
                avg_fitness_over_stag_dur = sum(avg_fitness_history[-max_stag_dur:]) / max_stag_dur
                non_stag_fitness = avg_fitness_history[-max_stag_dur] * non_stag_improv
                if avg_fitness_over_stag_dur < non_stag_fitness:
                    # Species stagnating. Remove it.
                    logging.debug("Removing species {} as stagnating for {} generations..."
                                  .format(species_id, max_stag_dur))
                    for genome_id in self.species_assignment[species_id]:
                        population.delete_genome(genome_id)
                        del self.genomes_adj_fitness[genome_id]
                    del self.species_assignment[species_id]
                    del self.species_avg_fitness_history[species_id]

        # Predetermine variables required for the creation of new genomes
        mutation_weights_randval = self.crossover_prob + self.mutation_weights_prob
        mutation_add_conn_randval = mutation_weights_randval + self.mutation_add_conn_prob
        pop_size = population.get_pop_size()
        species_adj_fitness_average = dict()
        for species_id, species_genome_ids in self.species_assignment.items():
            adj_fitness_average = 0
            for species_genome_id in species_genome_ids:
                adj_fitness_average += self.genomes_adj_fitness[species_genome_id]
            species_adj_fitness_average[species_id] = adj_fitness_average
        total_adj_fitness_average = sum(species_adj_fitness_average.values())

        # Create new genomes through evolution
        for species_id, species_genome_ids in self.species_assignment.items():
            # Determine fraction of population suitable to be a parent according to 'reproducing_fraction' cfg parameter
            reproduction_cutoff_index = int(self.reproducing_fraction * len(species_genome_ids))
            if reproduction_cutoff_index == 0:
                reproduction_cutoff_index = 1
            parent_genome_ids = species_genome_ids[:reproduction_cutoff_index]

            # Determine number of allotted offspring for the species and subtract 1 due to NEATs genome elitism of 1
            allotted_offspring = round(species_adj_fitness_average[species_id] * pop_size /
                                       total_adj_fitness_average) - 1

            for _ in range(allotted_offspring):
                parent_genome = population.get_genome(choice(parent_genome_ids))

                # Create random value and choose either one of the crossovers/mutations
                evolution_choice_val = random()
                if evolution_choice_val < self.crossover_prob:
                    # Out of simplicity currently possible that both parent genomes are the same
                    second_parent_genome = population.get_genome(choice(parent_genome_ids))
                    new_genome_id, new_genome = self._create_crossed_over_genome(parent_genome, second_parent_genome)
                elif evolution_choice_val < mutation_weights_randval:
                    new_genome_id, new_genome = self._create_mutated_weights_genome(parent_genome)
                elif evolution_choice_val < mutation_add_conn_randval:
                    new_genome_id, new_genome = self._create_added_conn_genome(parent_genome)
                else:
                    new_genome_id, new_genome = self._create_added_node_genome(parent_genome)
                # Add the newly created genome to the population immediately
                population.add_genome(new_genome_id, new_genome)

            # Delete all parent genomes of species as generations completely replace each other, though keep the species
            # champion as NEAT has genome elitism of 1
            for species_genome_id in species_genome_ids[1:]:
                population.delete_genome(species_genome_id)
                del self.genomes_adj_fitness[species_genome_id]
            del self.species_assignment[species_id][1:]

    def _create_crossed_over_genome(self, parent_genome_1, parent_genome_2) -> (int, DirectEncodingGenome):
        """
        Create a crossed over genome according to NEAT crossover illustration (since written specification in
        O Stanley's PhD thesis contradictory) by joining all disjoint and excess genes from both parents and choosing
        the parent gene randomly from either parent in case both parents possess the gene. Return that genome.
        :param parent_genome_1: DirectEncoding genome, parent genome that constitutes the basis for the mutation
        :param parent_genome_2: DirectEncoding genome, parent genome that constitutes the basis for the mutation
        :return: tuple of genome-id and its corresponding newly created DirectEncoding genome, which is a mutated
                 offspring from the supplied parent genome
        """
        genotype_1 = parent_genome_1.get_genotype()
        genotype_2 = parent_genome_2.get_genotype()
        existing_genes = set(genotype_1).union(set(genotype_2))

        new_genotype = dict()
        for gene_id in existing_genes:
            # If matching genes of both genotypes
            if gene_id in genotype_1 and gene_id in genotype_2:
                # Choose randomly from which parent the gene will be carried over
                if randint(0, 1):
                    new_genotype[gene_id] = deepcopy(genotype_1[gene_id])
                else:
                    new_genotype[gene_id] = deepcopy(genotype_2[gene_id])
            # If gene a excess or disjoint gene from genotype 1
            elif gene_id in genotype_1:
                new_genotype[gene_id] = deepcopy(genotype_1[gene_id])
            # If gene a excess or disjoint gene from genotype 2
            else:
                new_genotype[gene_id] = deepcopy(genotype_2[gene_id])

        return self.encoding.create_genome(new_genotype)

    def _create_mutated_weights_genome(self, parent_genome) -> (int, DirectEncodingGenome):
        """
        Create a mutated weights genome according to NEAT by adding to each chosen gene's conn_weight or bias a random
        value from a normal distribution with cfg specified mean and stddev. Only x percent (as specified via cfg
        parameter 'mutation_weights_fraction') of all gene's weights are actually mutated, to allow for a more fine-
        grained evolution. Return that genome.
        :param parent_genome: DirectEncoding genome, parent genome that constitutes the basis for the mutation
        :return: tuple of genome-id and its corresponding newly created DirectEncoding genome, which is a mutated
                 offspring from the supplied parent genome
        """
        new_genotype = deepcopy(parent_genome.get_genotype())
        gene_ids = tuple(new_genotype)

        for _ in range(int(len(new_genotype) * self.mutation_weights_fraction)):
            # Create weight to mutate with (as identical for both conn_weight and bias)
            mutation_weight = np.random.normal(loc=self.mutation_weights_mean, scale=self.mutation_weights_stddev)
            # Choose random gene to mutate
            mutated_gene_id = choice(gene_ids)
            # Identify type of gene and mutate its weight by going with a pythonic 'try and fail safely' approach
            try:
                new_genotype[mutated_gene_id].conn_weight += mutation_weight
            except AttributeError:
                new_genotype[mutated_gene_id].bias += mutation_weight
        return self.encoding.create_genome(new_genotype)

    def _create_added_conn_genome(self, parent_genome) -> (int, DirectEncodingGenome):
        """
        Create a added conn genome according to NEAT by randomly connecting two previously unconnected nodes. Return
        that genome. If parent genome is fully connected, return the parent genome.
        :param parent_genome: DirectEncoding genome, parent genome that constitutes the basis for the mutation
        :return: tuple of genome-id and its corresponding newly created DirectEncoding genome, which is a mutated
                 offspring from the supplied parent genome
        """
        new_genotype = deepcopy(parent_genome.get_genotype())
        topology_levels = parent_genome.get_topology_levels()

        # Record existing connections in genotype through pythonic 'try and fail safely' approach
        existing_genotype_conns = set()
        for gene in new_genotype.values():
            try:
                existing_genotype_conns.add((gene.conn_in, gene.conn_out))
            except AttributeError:
                pass

        # Convert possible_conn_ins to a list in order to shuffle it (which is not possible would I stick with an
        # Iterator over the set), which in turn is necessary to add the connection at a genuinely random place.
        max_index_conn_in = len(topology_levels) - 1
        possible_conn_ins = list(set.union(*topology_levels[:max_index_conn_in]))
        shuffle(possible_conn_ins)

        conn_added_flag = False
        for conn_in in possible_conn_ins:
            # Determine a list of (shuffled) conn_outs for the randomized conn_in
            for layer_index in range(max_index_conn_in):
                if conn_in in topology_levels[layer_index]:
                    min_index_conn_out = layer_index + 1
            possible_conn_outs = list(set.union(*topology_levels[min_index_conn_out:]))
            shuffle(possible_conn_outs)

            for conn_out in possible_conn_outs:
                if (conn_in, conn_out) not in existing_genotype_conns:
                    conn_weight = np.random.normal(loc=self.mutation_weights_mean, scale=self.mutation_weights_stddev)
                    new_gene_id, new_gene_conn = self.encoding.create_gene_connection(conn_in, conn_out, conn_weight)
                    new_genotype[new_gene_id] = new_gene_conn
                    conn_added_flag = True
                    break
            if conn_added_flag:
                break

        return self.encoding.create_genome(new_genotype)

    def _create_added_node_genome(self, parent_genome) -> (int, DirectEncodingGenome):
        """
        Create a added node genome accordng to NEAT by randomly splitting a connection. The connection is split by
        introducing a new node that has a connection to the old conn_in with a conn_weight of 1 and a connection to the
        old conn_out with the old conn_weight. Return that genome.
        :param parent_genome: DirectEncoding genome, parent genome that constitutes the basis for the mutation
        :return: tuple of genome-id and its corresponding newly created DirectEncoding genome, which is a mutated
                 offspring from the supplied parent genome
        """
        new_genotype = deepcopy(parent_genome.get_genotype())
        # Choose a random gene connection(!) from the genotype (and check that no gene node was accidentally chosen)
        genotype_gene_ids = tuple(new_genotype)
        while True:
            gene = new_genotype[choice(genotype_gene_ids)]
            if hasattr(gene, 'conn_weight'):
                break

        # Extract all required information from chosen gene to build a new node in between and then disable it (Not
        # remove it as per NEAT specification). If between the conn_in and conn_out has already been another node added
        # in another mutation, use the same node
        conn_in = gene.conn_in
        conn_out = gene.conn_out
        conn_weight = gene.conn_weight
        gene.set_enabled(False)
        if (conn_in, conn_out) in self.add_node_history:
            node = self.add_node_history[(conn_in, conn_out)]
        else:
            self.node_counter += 1
            self.add_node_history[(conn_in, conn_out)] = self.node_counter
            node = self.node_counter

        gene_node_id, gene_node = self.encoding.create_gene_node(node, 0, self.activation_hidden)
        gene_conn_in_node_id, gene_conn_in_node = self.encoding.create_gene_connection(conn_in, node, 1)
        gene_conn_node_out_id, gene_conn_node_out = self.encoding.create_gene_connection(node, conn_out, conn_weight)
        new_genotype[gene_node_id] = gene_node
        new_genotype[gene_conn_in_node_id] = gene_conn_in_node
        new_genotype[gene_conn_node_out_id] = gene_conn_node_out

        return self.encoding.create_genome(new_genotype)

    def evaluate_population(self, population, genome_eval_function):
        """
        Evaluate population by first evaluating each previously unevaluated genome on the genome_eval_function and
        saving its fitness. The population is the clustered and fitness sharing is applied according to the NEAT
        specification.
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        :param genome_eval_function: callable method that takes a genome as input and returns the fitness score
                                     corresponding to the genomes performance in the environment
        """
        genome_ids = population.get_genome_ids()
        for genome_id in genome_ids:
            genome = population.get_genome(genome_id)
            # Only evaluate genome fitness if it has not been evaluated before (as genome doesn't change)
            if genome.get_fitness() == 0:
                genome.set_fitness(genome_eval_function(genome))

        # Speciate population by first clustering it and then applying fitness sharing
        self._cluster_population(population, genome_ids)
        self._apply_fitness_sharing(population)

    def _cluster_population(self, population, genome_ids):
        """
        Cluster population by assigning each genome either to an existing species or to a new species, for which that
        genome will then become the representative. If a genome's distance to a species representative is below the
        distance threshold it is assigned to the species, whose species representative is closest.
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        :param genome_ids: list of all keys/genome_ids of the population's genomes
        """
        # Create a seperate dict that already contains each species representative, as they will be accessed often
        species_representatives = dict()
        species_representative_ids = set()
        for species_id in self.species_assignment:
            species_representative_id = self.species_assignment[species_id][0]
            species_representatives[species_id] = population.get_genome(species_representative_id)
            species_representative_ids.add(species_representative_id)

        # Assert that the chosen species clustering algorithm is 'threshold-fixed' as other not yet implemented
        assert self.species_clustering[0] == 'threshold-fixed'

        # Determine the distance of each genome to the representative genome of each species (first genome in the list
        # of species assigned genomes) and save it in the distance dict (key: species_id, value: genome's distance to
        # that the representative of that species)
        distance = dict()
        distance_threshold = self.species_clustering[1]
        for genome_id in genome_ids:
            # Skip evaluation of genome if it is already a representative
            if genome_id in species_representative_ids:
                continue

            genome = population.get_genome(genome_id)
            for species_id, species_representative in species_representatives.items():
                distance[species_id] = self._calculate_genome_distance(genome, species_representative)

            closest_species_id = min(distance, key=distance.get)
            if distance[closest_species_id] <= distance_threshold:
                # Assign genome to the closest existing species, as distance to other species not great enough to
                # warrant the creation of a new species
                self.species_assignment[closest_species_id].append(genome_id)
            else:
                # Genome is distant enough from any other species representative that the creation of a new species with
                # itself as the species representative is appropriate
                self.species_id_counter += 1
                self.species_assignment[self.species_id_counter] = [genome_id]
                self.species_avg_fitness_history[self.species_id_counter] = []
                species_representatives[self.species_id_counter] = population.get_genome(genome_id)

        # Since all clusters are created, sort the genomes of each species by their fitness. This will set the fittest
        # genome as the new genome representative as well as allow for easy determination of the fittest fraction that
        # is allowed to reproduce later.
        for species_id, species_genome_ids in self.species_assignment.items():
            sorted_species_genome_ids = sorted(species_genome_ids, key=lambda x: population.get_genome(x).get_fitness(),
                                               reverse=True)
            self.species_assignment[species_id] = sorted_species_genome_ids

    def _calculate_genome_distance(self, genome_1, genome_2) -> float:
        """
        Calculate the distance between 2 DirectEncodingGenomes according to NEAT's genome distance formula:
        distance = (c1 * E)/N + (c2 * D)/N + c3 * W
        E: amount of excess genes between both genotypes
        D: amount of disjoint genes between both genotypes
        W: average weight difference of matching genes between both genotypes. For this, TFNE does not only consider the
           weight differences between connection weights, but also weight differences between node biases.
        N: length of the longer genotype
        c1: cfg specified coefficient adjusting the importance of excess gene distance
        c2: cfg specified coefficient adjusting the importance of disjoint gene distance
        c3: cfg specified coefficient adjusting the importance weight distance
        :param genome_1: DirectEncoding genome
        :param genome_2: DirectEncoding genome
        :return: Distance between the two supplied DirectEncoding genomes in terms of number of excess genes, number of
                 disjoint genes and avg weight difference for the matching genes
        """
        genotype_1 = genome_1.get_genotype()
        genotype_2 = genome_2.get_genotype()
        gene_ids_1 = set(genotype_1)
        gene_ids_2 = set(genotype_2)
        max_genotype_length = max(len(gene_ids_1), len(gene_ids_2))

        # Determine gene_id from which on out other genes count as excess or up to which other genes count as disjoint
        max_gene_id_1 = max(gene_ids_1)
        max_gene_id_2 = max(gene_ids_2)
        excess_id_threshold = min(max_gene_id_1, max_gene_id_2)

        # Calculation of the first summand of the total distance, the excess gene distance. First determine excess genes
        if excess_id_threshold == max_gene_id_1:
            excess_genes = set()
            for id in gene_ids_2:
                if id > excess_id_threshold:
                    excess_genes.add(id)
        else:
            excess_genes = set()
            for id in gene_ids_1:
                if id > excess_id_threshold:
                    excess_genes.add(id)
        excess_gene_distance = self.distance_excess_c1 * len(excess_genes) / max_genotype_length

        # Calculation of the second summand of the total distance, the disjoint gene distance
        disjoint_genes_length = len((gene_ids_1.symmetric_difference(gene_ids_2)).difference(excess_genes))
        disjoint_gene_distance = self.distance_disjoint_c2 * disjoint_genes_length / max_genotype_length

        # Calculation of the third summand of the total distance, the average weight differences of matching genes
        matching_gene_ids = gene_ids_1.intersection(gene_ids_2)
        total_weight_difference = 0
        for gene_id in matching_gene_ids:
            try:
                total_weight_difference += np.abs(np.subtract(genotype_1[gene_id].conn_weight,
                                                              genotype_2[gene_id].conn_weight))
            except AttributeError:
                total_weight_difference += np.abs(np.subtract(genotype_1[gene_id].bias,
                                                              genotype_2[gene_id].bias))
        matching_gene_distance = self.distance_weight_c3 * total_weight_difference / len(matching_gene_ids)

        return excess_gene_distance + disjoint_gene_distance + matching_gene_distance

    def _apply_fitness_sharing(self, population):
        """
        Calculate the adjusted fitness score of each genome and save it in an internal dict as it is required for the
        calculation of the allotted offspring for each species. Also log the species avg fitness as all calculations are
        performed here anyway
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        """
        for species_id, species_genome_ids in self.species_assignment.items():
            species_size = len(species_genome_ids)
            fitness_sum = 0
            for genome_id in species_genome_ids:
                genome_fitness = population.get_genome(genome_id).get_fitness()
                fitness_sum += genome_fitness
                # Calculate the genome's adjusted fitness to 3 decimal places and save it internally
                adjusted_fitness = round(genome_fitness / species_size, 3)
                self.genomes_adj_fitness[genome_id] = adjusted_fitness
            species_avg_fitness = round(fitness_sum / species_size, 3)
            self.species_avg_fitness_history[species_id].append(species_avg_fitness)

    def summarize_population(self, population):
        """
        Output a summary of the population to logging.info, summarizing the status of the whole population as well as
        the status of each species in particular. Also output the string representation fo each genome to logging.debug.
        :param population: Population object of the TFNE framework, serving as an abstraction for the genome collection
        """
        # Summarize whole population:
        generation = population.get_generation_counter()
        best_fitness = population.get_best_genome().get_fitness()
        avg_fitness = population.get_average_fitness()
        pop_size = population.get_pop_size()
        species_count = len(self.species_assignment)
        logging.info("#### GENERATION: {:>4} ## BEST_FITNESS: {:>8} ## AVERAGE_FITNESS: {:>8} "
                     "## POP_SIZE: {:>4} ## SPECIES_COUNT: {:>4} ####"
                     .format(generation, best_fitness, avg_fitness, pop_size, species_count))

        # Summarize each species and its genomes seperately
        for species_id, species_genome_ids in self.species_assignment.items():
            species_best_fitness = population.get_genome(species_genome_ids[0]).get_fitness()
            species_avg_fitness = self.species_avg_fitness_history[species_id][-1]
            species_size = len(species_genome_ids)
            logging.info("---- SPECIES_ID: {:>4} -- SPECIES_BEST_FITNESS: {:>4} "
                         "-- SPECIES_AVG_FITNESS: {:>4} -- SPECIES_SIZE: {:>4} ----"
                         .format(species_id, species_best_fitness, species_avg_fitness, species_size))
            for genome_id in species_genome_ids:
                logging.debug(population.get_genome(genome_id))

    def get_species_report(self) -> dict:
        """
        Create a species report dict listing all currently present species ids as keys and assigning them the size of
        their species as value.
        :return: dict, containing said species report assigning species_id to size of species
        """
        species_report = dict()
        for species_id, species_genome_ids in self.species_assignment.items():
            species_report[species_id] = len(species_genome_ids)
        return species_report
