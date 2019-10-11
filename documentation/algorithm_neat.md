## TFNE Documentation for Algorithm 'Neuroevolution of Augmenting Topologies' (NEAT) ##

--------------------------------------------------------------------------------

#### Specification ####

NEAT is a benchmark neuroevolution algorithm evolving direct encoded genotypes 
that was developed in 2002 by Kenneth O. Stanley and Risto Miikkulainen [1,2,3].
Its most discerning features were the introduction of Historical-Markings, 
allowing for lossless crossover of different genotypes as well as strict 
speciation, protecting topological innovations. TFNE NEAT implements the
algorithm exactly as specified [1,3], with the listed peculiarities and
constraints listed below as NEAT's specification does allow for some leeway

[1] http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
[2] http://nn.cs.utexas.edu/downloads/papers/stanley.cec02.pdf
[3] http://nn.cs.utexas.edu/downloads/papers/stanley.phd04.pdf



--------------------------------------------------------------------------------

#### Configuration File (cfg) Parameter Documentation ####

* reproducing_fraction    # float; top fraction of the species that is elligible
                            to be selected as a parent for crossover/mutation.

* crossover_prob          # float; Probability a genome will be crossed over
                            with a random other one during the evolution.
                            Constraint: crossover_prob + mutation_weights_prob +
                            mutation_add_conn_prob + mutation_add_node_prob = 1

* mutation_weights_prob   # float; Probability a genomes weights will be mutated
                            during the evolution.
                            Constraint: crossover_prob + mutation_weights_prob +
                            mutation_add_conn_prob + mutation_add_node_prob = 1

* mutation_add_conn_prob  # float; Probability a random connection gene will be
                            added to the genome during the evolution.
                            Constraint: crossover_prob + mutation_weights_prob +
                            mutation_add_conn_prob + mutation_add_node_prob = 1

* mutation_add_node_prob  # float; Probability a random node gene will be added
                            to the genome during the evolution.
                            Constraint: crossover_prob + mutation_weights_prob +
                            mutation_add_conn_prob + mutation_add_node_prob = 1

* mutation_weights_mean   # float; Mean of the normal standard distribution from 
                            which the values are taken that are to be added to 
                            the gene weights.

* mutation_weights_stddev # float; Standard deviation of the normal standard 
                            distribution from which the values are taken that 
                            are to be added to the gene weights.

* distance_excess_c1      # float; Value of coefficient c1 (weighing the 
                            importance of excess genes) used in NEAT's formula 
                            to determine the distance between two genes.

* distance_disjoint_c2    # float; Value of coefficient c2 (weighing the 
                            importance of disjoint genes) used in NEAT's formula
                            to determine the distance between two genes.

* distance_weight_c3      # float; Value of coefficient c3 (weighing the 
                            importance of the average weight difference between 
                            matching genes) used in NEAT's formula to determine 
                            the distance between two genes.

* activation_hidden       # string; Activation function the algorithm will use 
                            for all created hidden nodes.

* activation_output       # string; Activation function the algorithm will use 
                            for all created output nodes.

* species_elitism         # int; Elitism of the species when determining species
                            stagnation. The amount of species specified in this
                            variable will always persist, even when stagnating.

* species_max_stagnation  # int, float; First parameter specifies the duration 
                            over which stagnation is considered. Second
                            parameter specifies the minimum percentual 
                            improvement of the species average fitness over the 
                            specified duration for a species to not be 
                            considered stagnating.

* species_clustering      # string, float; First parameter specifies the
                            method of species-clustering (Only 'threshold-fixed'
                            supported at the moment, but 'threshold-dynamic' is
                            planned). Second parameter specifies the minimum
                            distance between two genomes for them to be 
                            considered different enough to be assigned to
                            different clusters/species.



--------------------------------------------------------------------------------

#### Peculiarities of TFNE Implementation ####

As NEAT allows for some leeway in its implementation, is here a (non-exhaustive)
list of descriptions on how TFNE NEAT implements the aspects of NEAT that allow
leeway:

* NEAT does not dictate activation functions or node biases in gene nodes. TFNE 
  NEAT allows for this.

* NEAT does not dictate the method of gene weight initialization, just indicates
  that the initial population only differs in their weights. TFNE initializes 
  connection weights and node biases to 0 and mutates the connection weights 
  (not the node biases) once by adding values from the random normal 
  distribution with mean and std-dev set by cfg.

* NEAT does not dictate if a parent genome can go through multiple crossovers/
  mutations in a single evolution (e.g. mutation of weights and then addition of
  node). TFNE NEAT performs exactly one crossover/mutation (crossover, weight-
  mutaton, add-node-mutation, add-conn-mutation) with the parent genome in each 
  generation.

* NEAT does not dictate if genomes with mutated weights are considered 'new'
  genomes. TFNE NEAT does consider them as new genomes and assigns a new
  genome-id.

* NEAT does not dictate which weights exactly are to be mutated when mutating 
  weights. TFNE NEAT mutates all connection weights and all node biases.

* NEAT dictates a fixed species_elitism of 0, allowing the population to go 
  extinct. TFNE NEAT allows to set species_elitism and saves the specified 
  amount of best performing species even when stagnating.



--------------------------------------------------------------------------------

#### Constraints of TFNE Implementation ####

Shortcomings of the current implementation of TFNE NEAT:

* NEAT allows for recurrent networks to emerge. TFNE NEAT does as of yet only 
  allow for feed-forward networks as the underlying direct encoding as of yet 
  only supports feed-forward networks.

* TFNE NEAT has not yet implemented dynamic threshold species clustering.



