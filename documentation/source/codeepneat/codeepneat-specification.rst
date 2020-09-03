..
   Define line break and bullet command for this section, as necessary for
   properly formatted list in tables

.. |br| raw:: html

   <br />

.. |bullet| unicode:: \u2022

CoDeepNEAT Specification
========================

.. note:: This documentation solely lists the algorithm & encoding specifications without concerning itself with the validity or potential of the specific choices that make up the CoDeepNEAT method.

.. warning::  This documentation outlines the CoDeepNEAT algorithm & encoding specifications as understood by the TFNE project. While the TFNE project aims to stay as close as possible to the original specification, does it also aim to be a superset of the configuration options of the original specification. This specification also concretizes the algorithm specification if the original specification is too vague and no code was supplied. If you find an issue with the specification or the implementation details please contact tfne@paulpauls.de. Thank you.


--------------------------------------------------------------------------------

CoDeepNEAT Encoding
-------------------

The genotype of a CoDeepNEAT genome is made up of 2 essential parts. The first part is the CoDeepNEAT blueprint, specifying the ANN topology and the training hyperparameters that will be associated with the genome. The second part is a collection of small fully functional deep neural networks, termed the CoDeepNEAT modules, that will replace the nodes in the blueprint specified ANN topology. It is important to understand these two essential parts, what they entail, their exact contribution to the final genome and how exactly they are evolved in order to fully understand the CoDeepNEAT encoding and resulting genomes.


CoDeepNEAT Blueprint
~~~~~~~~~~~~~~~~~~~~

+---------------------------------------------+----------------------------------+
| Blueprint genotype                          | |bullet| Blueprint graph |br|    |
|                                             | |bullet| Optimizer configuration |
+---------------------------------------------+----------------------------------+

A blueprint is the fundamental building block of a CoDeepNEAT genome, specifying the genome's basic ANN topology as well as all its hyperparameters that may be associated with that genome.

In TFNE the current extent of hyperparameters saved by blueprints is a full configuration of a Tensorflow optimizer, specifying all variables required for the training of the genome phenotype TF model. Additional hyperparameters such as possible training preprocessing operations can also be included, though are currently not part of TFNE's CoDeepNEAT. This Tensorflow optimizer configuration is the first part of the blueprint's genotype.

The second part of the blueprint genotype is the graph that is specifying the basic ANN topology. This graph will be referred to as the blueprint graph. The blueprint graph is a collection of node and connection *gene* instances. In TFNE, those node and connection gene classes are defined as listed below, demonstrating the extent of the information they contain. The purpose and functionality of these blueprint graph genes is very similar to the functionality of genome genes in the original NEAT algorithm [see `NEAT <../neat/neat-overview.html>`_], as they are adapted from those. The difference being that each node gene stores a module species and each connection gene merely indicates connections between nodes but not associated connection weights. As in NEAT is TFNE currently restricting the blueprint graph to representing a feedforward graph, though a later addition to support full recurrent graphs is planned.

.. code-block:: python

    class CoDeepNEATBlueprintNode:
        def __init__(self, gene_id, node, species):
            self.gene_id = gene_id
            self.node = node
            self.species = species

    class CoDeepNEATBlueprintConn:
        def __init__(self, gene_id, conn_start, conn_end, enabled=True):
            self.gene_id = gene_id
            self.conn_start = conn_start
            self.conn_end = conn_end
            self.enabled = enabled

        def set_enabled(self, enabled):
            self.enabled = enabled


Each gene is assigned an ID. This ID is not unique to each gene instance but unique to each configuration of (node, species) or (conn_start, conn_end) respectively. This behavior is important to adhere to the principle of *Historical Markings* [see `NEAT <../neat/neat-overview.html>`_]. The ``node`` value is graph-unique integer identifier for each node and is specified in ``conn_start`` and ``conn_end`` as the start- and endpoint of the connection. Each connection can be disabled through a mutation or crossover. The ``species`` value is the ID of an existing, non-extinct module species and is relevant for the later assembly of the final genome phenotype model, combining blueprint and modules.


CoDeepNEAT Module
~~~~~~~~~~~~~~~~~

+---------------------------------------------+--------------------------------+
| Module genotype                             | |bullet| Merge method |br|     |
|                                             | |bullet| Module DNN parameters |
+---------------------------------------------+--------------------------------+

A CoDeepNEAT module is a class of small deep neural networks that can take on only limited complexity. The ANN topology as well as the parameters of the ANN layers are determined through a uniform set of parameters serving as the module genotype. However, since the set of parameters for a module instance is uniform and bounded, does this prevent the topology to become overly complex as only limited information can be stored in a CoDeepNEAT module instance. On the other hand does this allow for a direct comparison of module parameters as each module instance stores values for each module parameter.

A CoDeepNEAT module is obviously a very general concept and its specifics are highly dependent on the concrete implementation. A simple example module is the pre-implemented ``DenseDropout`` module [see `CoDeepNEAT Modules <./codeepneat-modules.html>`_], whose genotype storing has been implemented in TFNE as listed below. The module stores multiple parameters for the initial dense layer, a flag determining the presence of an optional subsequent dropout layer as well as parameters for that subsequent dropout layer. This simple class of module can only represent 2 possible ANN topologies, though it can potentially represent any valid parameter combination for the layer configuration.

.. code-block:: python

    class CoDeepNEATModuleDenseDropout(CoDeepNEATModuleBase):

        ...

        # Register the module parameters
        self.merge_method = merge_method
        self.units = units
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.dropout_flag = dropout_flag
        self.dropout_rate = dropout_rate


The uniformity of module parameters mentioned above simplifies evolutionary operations such as speciation, mutation and crossover. More importantly however does the resulting limited complexity resemble the concept of CoDeepNEAT in that it aims to evolve effective small DNNs in a repetitive network topology quickly in order to exploit the same repetitive structure in the problem environment. These repetitive and deep structures are seen in many successful recent DNNs.

The module genotype also requires a specification of a specific merge method as well as a method for downsampling input for this module. Both methods become relevant when combining blueprint and modules in the genome assembly. As the creation of an appropriate downsampling layer can be very complex is this functionality coded into the module itself in TFNE and is therefore not part of the genotype.

The section `CoDeepNEAT Modules <./codeepneat-modules.html>`_ introduces multiple pre-implemented modules provided by TFNE.


CoDeepNEAT Genome
~~~~~~~~~~~~~~~~~

+---------------------------------------------+-----------------------------------------------------------+
| Genome genotype                             | |bullet| Blueprint |br|                                   |
|                                             | |bullet| 1 Module for each Mod species present in BP |br| |
|                                             | |bullet| Output layers                                    |
+---------------------------------------------+-----------------------------------------------------------+

A CoDeepNEAT genome is comprised of 1 blueprint and 1 module for each module species that is present in the nodes of the BP graph. The genome adopts all hyperparameters from the associated blueprint, which in TFNE implies the configuration of the TF optimizer used in the eventual model training.

The phenotype of the genome is a Tensorflow model that is assembled by combining the blueprint graph and modules. The basic topology of the phenotype model is dictated by the graph represented in blueprint graph. TFNE currently only supporting feedforward graphs, though we hope to implement recurrent graphs soon. Each node in that blueprint graph will be replaced by a module, depending on the ``species`` value of the node. As modules themselves are small DNNs will the resulting graph be a full DNN consisting of multiple small DNNs that are connected to each other. If a module has multiple inputs will the inputs be merged according to the ``merge_method`` genotype value of the module. If a module has an input with mismatching dimensions will the input be downsampled through the specific downsampling method associated with the module type, which in TFNE can be accessed through ``create_downsampling_layer(...)``. The genotype graph is fully assembled when appending a predefined set of output layers to the final layer of the evolved graph, in order to conform with the required output of the problem environment.

The modules that are chosen to replace the graph nodes based on their ``species`` value, are selected as follows: Each ``species`` value in the nodes of the blueprint graph is identifying an existing, non-extinct module species (a species is a cluster of similar members, see `CoDeepNEAT Evolution <./codeepneat-specification.html#evolution>`_ below). For each ``species`` value that is present in the blueprint graph, select one specific module from that identified module species. The created association between ``species`` ID value and specific module is the above mentioned part of the genome genotype. In the phenotype TF model assembly replace each node with the same corresponding specific module. This way beneficial topological structure and parametrized layers are replicated throughout the final TF model in order to exploit the same repetitive structure in the problem environment.

To summarize is the exact process of translating the genome genotype into the phenotype model illustrated below:

.. figure:: ../illustrations/codeepneat_genome_assembly_illustration.svg
   :align: center

   Illustration of the Assembly of a CoDeepNEAT Genome from Blueprint and Modules


--------------------------------------------------------------------------------

CoDeepNEAT Algorithm
--------------------

Unlike traditional neuroevolution algorithms does the CoDeepNEAT algorithm not operate on and evolve genomes directly, but instead primarily operates on blueprints and modules. Genomes are only assembled during the evaluation in order to determine the fitness of the associated blueprints and modules.


Initialization
~~~~~~~~~~~~~~

**see CoDeepNEAT.initialize_population(...)**

CoDeepNEAT initializes a minimal population as it has been modeled after NEAT, an additive neuroevolution algorithm. The initialization is therefore very simple. All modules of the population are initialized with random parameters and assigned to a single species.

All blueprints are initialized with a minimal graph of 2 nodes and a connection. The first node is node 1, serving as a special, non mutateable, input layer. The second node is node 2, serving as the output layer and being assigned a ``species`` value identifying the single species ID all initialized modules have been assigned to. The hyperparameters of all blueprints are initialized with random parameters. As done for modules will all blueprints of the population be assigned to a single initial blueprint species.


Evaluation
~~~~~~~~~~

**[see CoDeepNEAT.evaluate_population(...)]**

The CoDeepNEAT population is evaluated by assembling genomes from the population of blueprints and modules, evaluating those genomes and then transferring the achieved genome fitness back to their blueprint and module components.

For each blueprint in the population the algorithm assembles a predefined number of genomes that take that blueprint as their base. For each of these genomes that are to be assembled, specific modules of the referenced blueprint graph node species are chosen randomly from the module species. That blueprint, randomly chosen modules of all referenced module species as well as the constant set of output layers constitute a full genome genotype and generate a phenotype TF model according to the `genome encoding <./codeepneat-specification.html#codeepneat-genome>`_ above. The assembled genome is then applied to the evaluation environment and assigned the resulting fitness score.

If due to the random choice of modules for the blueprint graph an invalid TF model is generated from the genome genotype, the assembled genome is assigned a fitness score of 0. As the evolutionary process evolves blueprints and modules separately is it impossible to guarantee a genotype that results in a valid TF model when both blueprints and modules are paired randomly and without knowledge of that pairing during evolution.

The fitness value of the blueprints and modules is calculated after all genomes of the generation have been generated and evaluated. The fitness value of both blueprints and modules is the average fitness value of all genomes in which the respective blueprint or module was involved in.


Evolution
~~~~~~~~~

**[see CoDeepNEAT.evolve_population(...)]**

Evolving the CoDeepNEAT population can be divided into three major phases. First, the CoDeepNEAT population needs to be *selected*, which means that species and members of the population deemed fit enough to create offspring are selected while the rest of the population is erased. The second phase is the actual evolution, during which the parental members of the generation are mutated and crossed over in order to create novel offspring with beneficial features. The third and last phase during the evolution of a generation is the speciation of the population. The speciation clusters the members of the population in similar groups in order to identify traits and features, determine if those traits and features are beneficial and if applicable, facilitate the spread of those features or remove them from the population. The evolution of NEAT-like neuroevolution algorithms is guided through this speciation. TFNE currently supports three speciation methods for both modules and blueprints respectively, which are based on speciation methods from the original NEAT algorithm, though which have not been explicitly defined in the original research paper.

Since the different methods of speciation are very complex and take on an important role in NEAT-like neuroevolution algorithms is this specification of the CoDeepNEAT evolution subdivided into the specification of the actual mutation and crossover of the modules and blueprints as well as into the specification of the different speciation methods for them.


Module Mutation & Crossover
"""""""""""""""""""""""""""

The actual mutation and crossover phase for modules is very simple. As during the preceding selection phase all eligible parents for offspring have been determined and the number of offspring for each species has been calculated. New modules are created by conforming to those determined parameters and mutate / crossover the intended amount of parents until the determined amount of offspring for each species has been reached. New modules are not automatically assigned the same species as their parents but are to be assigned to a species independently in the following speciation phase. In TFNE, if the ``mod_spec_reinit_extinct`` parameter has been set to true will the amount of modules belonging to species that went extinct in the preceding phase be reinitialized and treated as regular offspring that will be speciated in the following phase.

**Mutation** - Mutation for modules is the simple perturbation of the parameters of the parent module. The extent and manner in which this takes place is left up to the concrete implementation of module class. TFNE's pre-implemented modules perturb the parent module's parameters during mutation by selecting the offspring parameter from a normal distribution with the parent parameter value as the *mean* and the size of the standard distribution set via config. Mutating categorical parameters is done by randomly choosing a new value. TFNE also supports a config parameter (``mod_max_mutation``) that specifies the maximum degree in percent to which the parent parameters can be mutated.

**Crossover** - Crossover for modules is again left up to the concrete module class implementation. In TFNE pre-implemented modules, crossover is performed by averaging out the sortable parameters of both parent modules while categorical parameters are carried over from the fitter parent module.


Blueprint Mutation & Crossover
""""""""""""""""""""""""""""""

The mutation and crossover phase for blueprints is very similar to that of modules, with the exception of having different explicit mutation and crossover operations and an extra constraint regarding the extinction of module species. The first step of the mutation and crossover phase for blueprints is the check of all parental blueprints if their blueprint graphs contain references to module species (in the ``species`` value of the nodes) that are going extinct during this generation's evolution. If so, the parent's blueprint graph is mutated by replacing all references to extinct module species with references to randomly chosen non-extinct module species. The resulting mutated blueprint is then kept as a potential parent instead of the non-valid blueprint.

The rest of the mutation and crossover phase for blueprints is identical to that of modules. New offspring for each species is generated according to the predetermined amount. The type of mutation or crossover through which the offspring is generated is determined via percentage chance set via the config. Reinitialized blueprints will be generated if the ``bp_spec_reinit_extinct`` config parameter has been enabled. All generated offspring will be speciated in the following phase.

**Add Connection Mutation** - This mutation adds one or multiple connections to the blueprint graph. Activating a disabled connection also counts as an added connection. The connection start and end nodes are chosen randomly. In TFNE, the amount of connections to add to the blueprint graph is determined by the ``bp_max_mutation`` config value, though at least 1.

**Add Node Mutation** - This mutation adds one or multiple nodes to the blueprint graph. Nodes are added by placing them *in the middle* of existing and enabled connections. The chosen connection is disabled and 1 node and 2 connection genes are added. The ``species`` value of the new node is chosen randomly. In TFNE, the amount of nodes to add to the blueprint graph is determined by the ``bp_max_mutation`` config value, though at least 1.

**Remove Connection Mutation** - This mutation removes one or multiple connections from the blueprint graph. Connections are removed randomly, though only connections that are not the last incoming or outgoing connections are considered (as not to desert a node and therefore also effectively remove a node). In TFNE, the amount of connections to remove from the blueprint graph is determined by the ``bp_max_mutation`` config value, though removing 0 connections due to none being available is valid. This mutation was not included in the original specification, as NEAT-like neuroevolution algorithms are exclusively additive.

**Remove Node Mutation** - This mutation removes one or multiple nodes from the blueprint graph. Nodes that are to be removed are chosen randomly, though the input and output node are unavailable. All connections interacting with the removed node are also removed and replaced by connections between all incoming nodes to all outgoing nodes of the node to be removed. In TFNE, the amount of nodes to remove from the blueprint graph is determined by the ``bp_max_mutation`` config value, though removing 0 nodes due to none being available is valid. This mutation was not included in the original specification, as NEAT-like neuroevolution algorithms are exclusively additive.

**Node Species Mutation** - This mutation changes one or multiple node ``species`` values of the blueprint graph. Nodes that are to be mutated are chosen randomly. The new ``species`` value of mutated nodes is chosen randomly from all existing, non-extinct module species. In TFNE, the amount of nodes to mutate in the blueprint graph is determined by the ``bp_max_mutation`` config value, though at least 1.

**Optimizer Mutation** - This mutation changes one or multiple parameters of the blueprint associated optimizer. Categorical parameters are mutated by choosing randomly from all available values. Sortable parameters are mutated by choosing a value from a normal distribution with the parent parameter value as the *mean* and the size of the standard distribution set via config. In TFNE, the amount of optimizer parameters to mutate is determined by the ``bp_max_mutation`` config value, though at least 1.

**Crossover** - Crossover combines the blueprint graphs of both parent blueprints and carries over the hyperparameters of the fitter parent blueprint to create a new offspring. The blueprint graphs are combined by merging all genes of both blueprint graphs. If a gene, identified by its ID, occurs in both parent blueprint graphs, choose randomly. This crossover can create recurrent blueprint graphs In TFNE, as the architecture currently only supports feedforward graphs, is the blueprint graph adjusted by removing recurrent nodes and connections and connecting orphaned nodes (generated by removing recurrent nodes/connections) to the input and output node respectively.


Module Speciation Type: ``basic``
"""""""""""""""""""""""""""""""""

*Basic* module speciation essentially has the effect of there being only 1 species for each available module type. If only module type is set to be available via the evolution config, then all modules are assigned to the same species.

The offspring for each module species is determined during the selection phase. It is calculated by first determining the intended size of each module species post evolution and then subtracting the amount of *elite* modules from this intended size. Elite modules are the best modules of the population, which are also carried over to the next generation without change. The amount of elite modules carried over can be set via the configuration. The intended size of each module species is calculated as follows: Let f\ :sub:`i` \ be the average fitness of species i. Let p be the total module population size. Let s\ :sub:`i` \ be the intended size of species i post evolution:

.. math::

  s_i = \frac{f_i}{\sum f_i} p


In words, the intended size of each module species post evolution corresponds to its share of the total average fitness multiplied by the total population size. Consideration of mathematical edge cases can be seen in the TFNE implementation.

Basic parental module selection is performed removing all modules from each species that do not pass the configuration specified reproduction threshold (see ``mod_spec_reprod_thres``). This reproduction threshold specifies a percentile threshold which potential parent modules have to pass. All modules that do not pass this percentile threshold are removed from the population.

Basic module speciation is extremely simple in that all newly generated modules will be assigned to the same species as the other modules of the same type.


Blueprint Speciation Type: ``basic``
""""""""""""""""""""""""""""""""""""

*Basic* blueprint speciation is very similar to its basic module speciation counterpart. However, since there are no multiple blueprint types possible, are all blueprints constantly assigned to species 1.

Offspring calculation for blueprints in the basic speciation theme is very simple in that corresponds to the blueprint population size minus the *elite* blueprints for this generation. Elite blueprints are determined analogue to elite modules in the basic speciation scheme in that they are the best blueprints of the generation that are carried over unchanged to the next generation and the amount can be set via the configuration.

Basic parental blueprint selection is performed analogue to basic module parent selection in that blueprints have to be in a higher percentile threshold within the species than specified in the configuration in order to be eligible as parents.

Basic blueprint speciation is extremely simple in that all newly generated blueprints will be assigned to species 1.


Module Speciation Type: ``param-distance-fixed``
""""""""""""""""""""""""""""""""""""""""""""""""

The *param-distance-fixed* speciation scheme clusters the modules into species according to the relative distance of the module parameters. If the parameter distance between a module and the rest of the species is above a certain threshold, does this module found a new species.

In parameter distance (both *fixed* and *dynamic*) speciation schemes is it necessary for module species to have a module representative. This species representative is required to have a parameter distance to all other modules in the species that is lower than the config specified threshold for justifying a new species. When calculating the parameter distance of a new module towards an existing module species is this performed by calculating the parameter distance between the module species representative and the new module. The first module that is founding a new module species is considered its species representative.

Since it is possible to generate new species in the parameter distance speciation schemes is it also possible for species to go extinct. In the selection phase, if the average fitness of species has been stagnating for a configuration specified amount of generations will this species be extinguished. In TFNE, a species is considered stagnating if the average fitness has not improved once over the specified timeframe. If the ``mod_spec_reinit_extinct`` configuration parameter has been enabled via the configuration will the population share that was intended for the extinguished species be reinitialized as new modules during the mutation & crossover phase. If the reinitialization parameter is disabled, will the population share that was intended for the extinguished species instead be divided among the persisting species.

During the selection phase and before parental modules for the generation are determined is it possible to rebase the module species representative in parameter distance speciation schemes if the configuration parameter ``mod_spec_rebase_repr`` is enabled. Rebasing the representative of each module species selects the fittest module of each species as the new module species representative that fulfills the representative requirement of a lower parameter distance to all other modules of the species than the configuration specified threshold.

The calculation of assigned offspring for each module species in parameter distance speciation is very similar to *basic* module speciation. The assigned offspring is calculated during the selection phase by first determining the intended size of each module species post evolution and then subtracting the amount of *elite* modules from this intended size. Elite modules are the best modules of the population as well as all module species representatives. These elite modules are carried over to the next generation without change. The amount of elite modules carried over can be set via the configuration. The intended size of each module species is calculated as follows: Let f\ :sub:`i` \ be the average fitness of species i. Let p be the total module population size. Let s\ :sub:`i` \ be the intended size of species i post evolution:

.. math::

  s_i = \frac{f_i}{\sum f_i} p


In words, the intended size of each module species post evolution corresponds to its share of the total average fitness multiplied by the total population size. Consideration of mathematical edge cases can be seen in the TFNE implementation.

Parental module selection in the parameter distance speciation scheme is identical to that of the basic speciation scheme. All modules that do not pass the configuration specified reproduction percentile threshold (see ``mod_spec_reprod_thres``) are not eligible as parents and are removed from the population.

The speciation phase of the parameter distance scheme functions as follows: Each new module determines the parameter distance to each existing species. The distance to a species is determined by calculating the distance to the species representative. If the new module has a distance below the configuration specified species distance (``mod_spec_distance``) to one or multiple species, assign the new module to the closest species, meaning the species with the lowest distance. If no existing module species has a distance lower than the config specified distance threshold, create a new species with the new module as its representative.

The exact calculation of the parameter distance between two modules is left up to the concrete module class implementation. In TFNE pre-implemented modules is the parameter distance between two modules calculated by first determining their congruence and then subtracting the actual congruence from the perfect congruence of 1.0. The congruence of continuous parameters is calculated by their relative distance. The congruence of categorical parameters is either 1.0 in case they are the same or it's 1 divided by the amount of possible values for that specific parameter.


Module Speciation Type: ``param-distance-dynamic``
""""""""""""""""""""""""""""""""""""""""""""""""""

*Dynamic parameter distance* speciation is identical to the *fixed* parameter distance speciation scheme, with 1 addition. The dynamic parameter distance speciation scheme has the additional configuration parameter ``mod_spec_species_count``, which specifies the desired amount of species present throughout the evolutionary process. The desired species count is achieved by adjusting the ``mod_spec_distance`` configuration parameter throughout the evolution at the end of the speciation phase.

The exact process of adjusting the module species distance parameter is adaptable. In TFNE, if the actual module species count is lower than the desired module species count then the ``mod_spec_distance`` parameter is decreased by 5% in order to make the process of creating new species more sensitive. If the actual module species count is higher than the desired module species count then the distance between all module species representatives is calculated and the ``mod_spec_distance`` parameter is replaced by the value that would eliminate all module species except for the desired amount of most distant species.


Blueprint Speciation Type: ``gene-overlap-fixed``
"""""""""""""""""""""""""""""""""""""""""""""""""

.. note:: The blueprint *gene overlap* (both *fixed* and *dynamic*) speciation scheme is nearly analogue identical to the module parameter distance speciation schemes, with the exception of the specification on how to calculate the distance between blueprints. For the sake of autonomy though is the whole speciation scheme repeated and slightly adapted. If you are familiar with the module parameter distance speciation scheme feel free to skip to the end at which the calculation of blueprint distance is specified.

The *gene-overlap-fixed* speciation scheme clusters the blueprints into species according to the percentual amount of overlapping genes in between blueprints. If the gene overlap between a blueprint and the rest of the species is above a certain threshold, does this blueprint found a new species.

In gene overlap (both *fixed* and *dynamic*) speciation schemes is it necessary for blueprint species to have a blueprint representative. This species representative is required to have a gene overlap distance to all other blueprints in the species that is lower than the config specified threshold for justifying a new species. When calculating the gene overlap distance of a new blueprint towards an existing blueprint species is this performed by calculating the gene overlap distance between the blueprint species representative and the new blueprint. The first blueprint that is founding a new blueprint species is considered its species representative.

Since it is possible to generate new species in the gene overlap speciation schemes is it also possible for species to go extinct. In the selection phase, if the average fitness of species has been stagnating for a configuration specified amount of generations will this species be extinguished. In TFNE, a species is considered stagnating if the average fitness has not improved once over the specified timeframe. If the ``bp_spec_reinit_extinct`` configuration parameter has been enabled via the configuration will the population share that was intended for the extinguished species be reinitialized as new blueprints during the mutation & crossover phase. If the reinitialization parameter is disabled, will the population share that was intended for the extinguished species instead be divided among the persisting species.

During the selection phase and before parental blueprints for the generation are determined is it possible to rebase the blueprint species representative in gene overlap speciation schemes if the configuration parameter ``bp_spec_rebase_repr`` is enabled. Rebasing the representative of each blueprint species selects the fittest blueprint of each species as the new blueprint species representative that fulfills the representative requirement of a lower gene overlap distance to all other blueprints of the species than the configuration specified threshold.

The assigned offspring for each blueprint species is calculated during the selection phase by first determining the intended size of each blueprint species post evolution and then subtracting the amount of *elite* blueprints from this intended size. Elite blueprints are the best blueprints of the population as well as all blueprint species representatives. These elite blueprints are carried over to the next generation without change. The amount of elite blueprints carried over can be set via the configuration. The intended size of each blueprint species is calculated as follows: Let f\ :sub:`i` \ be the average fitness of species i. Let p be the total blueprint population size. Let s\ :sub:`i` \ be the intended size of species i post evolution:

.. math::

  s_i = \frac{f_i}{\sum f_i} p


In words, the intended size of each blueprint species post evolution corresponds to its share of the total average fitness multiplied by the total population size. Consideration of mathematical edge cases can be seen in the TFNE implementation.

Parental blueprint selection in the gene overlap speciation scheme is identical to that of the basic speciation scheme. All blueprints that do not pass the configuration specified reproduction percentile threshold (see ``bp_spec_reprod_thres``) are not eligible as parents and are removed from the population.

The speciation phase of the gene overlap speciation scheme functions as follows: Each new blueprint determines the gene overlap distance to each existing species. The distance to a species is determined by calculating the distance to the species representative. If the new blueprint has a distance below the configuration specified species distance (``bp_spec_distance``) to one or multiple species, assign the new blueprint to the closest species, meaning the species with the lowest distance. If no existing blueprint species has a distance lower than the config specified distance threshold, create a new species with the new blueprint as its representative.

The gene overlap distance between two blueprints is calculated by first determining their congruence and then subtracting the actual congruence from the perfect congruence of 1.0. The congruence of two blueprints is calculated by determining the percentual overlap of each blueprint towards the other blueprint and then averaging out that percentual overlap.


Blueprint Speciation Type: ``gene-overlap-dynamic``
"""""""""""""""""""""""""""""""""""""""""""""""""""

*Dynamic gene overlap* speciation is identical to the *fixed* gene overlap speciation scheme, with 1 addition. The dynamic gene overlap speciation scheme has the additional configuration parameter ``bp_spec_species_count``, which specifies the desired amount of species present throughout the evolutionary process. The desired species count is achieved by adjusting the ``bp_spec_distance`` configuration parameter throughout the evolution at the end of the speciation phase.

The exact process of adjusting the blueprint species distance parameter is adaptable. In TFNE, if the actual blueprint species count is lower than the desired blueprint species count then the ``bp_spec_distance`` parameter is decreased by 5% in order to make the process of creating new species more sensitive. If the actual blueprint species count is higher than the desired blueprint species count then the distance between all blueprint species representatives is calculated and the ``bp_spec_distance`` parameter is replaced by the value that would eliminate all blueprint species except for the desired amount of most distant species.

