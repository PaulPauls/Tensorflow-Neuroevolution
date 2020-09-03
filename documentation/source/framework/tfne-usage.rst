Using TFNE
==========

Installation
------------

TFNE, being still in the beta development phase, changes often. To get the most current working release from the PyPI repository, use the following command:

.. code-block:: shell

    pip install tfne

To enable the rendering of genome graphs in code as well as in the TFNE Visualizer, make sure that the ``graphviz`` library is installed on the system (`Graphviz Installation <https://www.graphviz.org/download/>`_). For Ubuntu the following command will install graphviz from the package manager:

.. code-block:: shell

    sudo apt install graphviz

Import the TFNE library into your code by using the same name (``import tfne``). The TFNE Visualizer will be available as an executable script after installation under the name ``tfne_visualizer``.

--------------------------------------------------------------------------------

Neuroevolution
--------------

TFNE allows for the creation and improvement of artificial neural networks through a population-based optimization process called *neuroevolution*. In neuroevolution artificial neural networks are represented as genomes, encoding the properties such as topology, hyperparameters, layer configuration, node values, etc. The encoding of the genomes - the *genotypes* - can arbitrarily be mutated and combined. This allows for the emergence of new genomes that may represent novel ways to solve the optimization problem or the emergence of genomes with combined beneficial properties of two preceding genomes. The quality of the emerging genome is determined by calculating a *fitness* value that represents the degree to which the artificial neural network that is represented through the genome solves the problem environment. This allows for a relative comparison between genomes.

The optimization process itself can be guided through various means in order to produce the fittest genome. Speciation for example is a popular way of guiding the evolution by clustering genomes based on aspects such as topological similarity of the neural network or similarity in the problem-solving approach. This clustering allows the identification of generally beneficial or detrimental features and allows the optimization process to be guided by focusing its evolutionary development on clusters with beneficial features.


--------------------------------------------------------------------------------

Quick Start
-----------

The usage of TFNE is demonstrated in the ``examples/`` directory of the Github repository (`see here <https://github.com/PaulPauls/Tensorflow-Neuroevolution/tree/dev_tfne_v0.2/examples>`_). The examples employ multiple neuroevolution algorithms, problem environments and approaches to the problem and will be steadily extended in future TFNE releases. The basic approach to solving a problem in TFNE is as follows:

The first step is to decide on which NE algorithm to use and to create a complete configuration for all sections and options of the chosen algorithm. This configuration file needs to be in a format readable by python's ConfigParser class and is named ``config-file.cfg`` in the example below.

.. code-block:: python

    import tfne

    # Optional in case the ConfigParser object is created manually
    config = tfne.parse_configuration('./config-file.cfg')

    ne_algorithm = tfne.algorithms.CoDeepNEAT(config)


The next step is the instantiation of the problem environment the chosen neuroevolution algorithm should be applied to. Depending on if the TF model represented through the NE algorithm's genomes should first be trained before assigning the genome a fitness value, set the ``weight_training`` parameter of the chosen problem environment. Some problem environments do not allow for weight-training evaluation while some NE algorithms do not allow for non-weight-training evaluation. Here we are using the CIFAR10 problem environment.

.. code-block:: python

    environment = tfne.environments.CIFAR10nvironment(weight_training=True,
                                                      config=config,
                                                      verbosity=0)

    # Alternative, though identical, creation of the environment specifying parameters explicitly
    # instead of through the config
    environment_alt = tfne.environments.CIFAR10nvironment(weight_training=True,
                                                          verbosity=0,
                                                          epochs=8
                                                          batch_size=None)


The instantiated NE algorithm and evaluation environment are then handed over to the driving force of the evolutionary process - the ``EvolutionEngine``. The EvolutionEngine class prepares the start of the evolution and furthermore takes on the parameters for ``backup_dir_path``, ``max_generations`` and ``max_fitness``. The ``backup_dir_path`` parameter specifies the directory as string to which the evolution state backups should be saved to in order to postanalyze or visualize the evolution later. The parameters ``max_gnerations`` and ``max_fitness`` specify abort conditions for evolution.

.. code-block:: python

    engine = tfne.EvolutionEngine(ne_algorithm=ne_algorithm,
                                  environment=environment,
                                  backup_dir_path='./',
                                  max_generations=64,
                                  max_fitness=100.0)


The evolutionary process however is not started until calling the ``train()`` function of the set up evolutionary engine. This function returns the best genome - and therefore TF model - as judged by the evaluation environment the evolutionary process could produce.

.. code-block:: python

    best_genome = engine.train()
    print("Best genome returned by evolution:\n")
    print(best_genome)


These few and simple lines summarize the basic usage of TFNE. However, the underlying NE algorithms and environments can be extensively configured resulting in significantly different evolutionary processes. The extent to which the NE algorithms and environments can be configured and how exactly they operate is presented in their respective sections of this documentation.

TFNE however also serves as a prototyping platform for custom neuroevolution algorithms, populations, encodings and environments. The `TFNE Architecture & Customization <./tfne-customization.html>`_ section elaborates on this functionality.


--------------------------------------------------------------------------------

Serialization & Deserialization
-------------------------------

All pre-implemented NE algorithms in the TFNE framework serialize and save the state of the evolution in each generation as json files. These backups serve as the input for the `TFNE Visualizer <./tfne-visualizer.html>`_ or they can serve as initial states for the NE algorithms in case the evolution should be continued from a certain generation onward with a different algorithm configuration (though necessarily the same evaluation environment).

.. code-block:: python

    import tfne

    config = tfne.parse_configuration('./config-file.cfg')

    # Supply path to a backup serving as the initial state
    ne_algorithm = tfne.algorithms.CoDeepNEAT(config,
                                              initial_state_file_path='./tfne_state_backup_gen_15.json')


Serialization and deserialization is also possible for single genomes, e.g. in case the best genome of the evolution should be saved and deserialized later.

.. code-block:: python

    # Train and save best genome
    best_genome = engine.train()
    best_genome_file_path = best_genome.save_genotype(save_dir_path='./')

    # Load the serialized best genome and get the encoded TF model
    loaded_genome = tfne.deserialization.load_genome(best_genome_file_path)
    tf_model = loaded_genome.get_model()

    # Alternatively, it is also possible to save the TF model directly
    best_genome.save_model(file_path='./best_genome_model/')


--------------------------------------------------------------------------------

Projects Using TFNE
-------------------

* TBD


