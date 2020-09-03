## The Tensorflow-Neuroevolution Framework ##

<p align="center">
  <img src="./documentation/source/illustrations/tfne_logo.svg" width="40%" alt="TFNE Logo"/>
</p>

**Version 0.21.0**

![Python version req](https://img.shields.io/badge/python-v3.7%2B-informational)
[![PyPI version](https://badge.fury.io/py/tfne.svg)](https://badge.fury.io/py/tfne)
[![Documentation Status](https://readthedocs.org/projects/tfne/badge/?version=latest)](https://tfne.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/PaulPauls/Tensorflow-Neuroevolution/branch/master/graph/badge.svg)](https://codecov.io/gh/PaulPauls/Tensorflow-Neuroevolution)

The Tensorflow-Neuroevolution framework [abbr. TFNE] is a modular and high-performant prototyping platform for modern neuroevolution algorithms realized with Tensorflow 2.x. The framework implements already a variety of modern neuroevolution algorithms that are documented in detail in the extensive TFNE documentation and which are demonstrated in a multitude of examples. While the framework itself is optimized for high performance does the architecture design focus on maintainability, modularity and extendability by separating the main aspects of neuroevolution schemes - the problem environment, the genome encoding, the algorithm's population, and the neuroevolution algorithm itself.

All pre-implemented algorithms and genome encodings make heavy use Tensorflow and its internal optimization mechanisms. All pre-implemented encodings convert the genome genotype to a Tensorflow model phenotype through the usage of the Tensorflow keras functional API. This allows for high performance of the Tensorflow model phenotype as well as full compatibility with the rest of the Tensorflow ecosystem.

The framework is currently in a public development stage. The following modern neuroevolution algorithms are pre-implemented:

* CoDeepNEAT [[paper](https://arxiv.org/abs/1703.00548) | [doc](https://tfne.readthedocs.io/en/latest/codeepneat/codeepneat-overview.html) | [code](https://github.com/PaulPauls/Tensorflow-Neuroevolution/blob/master/tfne/algorithms/codeepneat/codeepneat.py)]
* NEAT (only in v0.1) [[paper](http://nn.cs.utexas.edu/keyword?stanley:phd04) | doc | code]

To demonstrate the capabilities of the neuroevolution algorithms and TFNE in particular are the following problem environments pre-implemented.

* XOR problem [[doc](https://tfne.readthedocs.io/en/latest/environments/xor-environment.html) | [code](https://github.com/PaulPauls/Tensorflow-Neuroevolution/blob/master/tfne/environments/xor_environment.py)]
* MNIST dataset [[doc](https://tfne.readthedocs.io/en/latest/environments/mnist-environment.html) | [code](https://github.com/PaulPauls/Tensorflow-Neuroevolution/blob/master/tfne/environments/mnist_environment.py)]
* CIFAR10 dataset [[doc](https://tfne.readthedocs.io/en/latest/environments/cifar10-environment.html) | [code](https://github.com/PaulPauls/Tensorflow-Neuroevolution/blob/master/tfne/environments/cifar10_environment.py)]

Both, the available algorithms as well as the available problem environments will be extensively updated for the final stable release. See [Roadmap to Stable](https://github.com/PaulPauls/Tensorflow-Neuroevolution/#roadmap-to-stable) below.


--------------------------------------------------------------------------------

### Installation ###

TFNE, being still in the beta development phase, changes often. To get the most current working release from the PyPI repository, use the following command:

``` bash
    pip install tfne
```

To enable the rendering of genome graphs in code as well as in the TFNE Visualizer, make sure that the `graphviz` library is installed on the system ([Graphviz Installation](https://www.graphviz.org/download/)). For Ubuntu the following command will install graphviz from the package manager:

``` bash
    sudo apt install graphviz
```

Import the TFNE library into your code by using the same name (`import tfne`). The TFNE Visualizer will be available as an executable script after installation under the name `tfne_visualizer`.


--------------------------------------------------------------------------------

### Usage ###

The usage of TFNE is demonstrated in the ``examples/`` directory of the Github repository (see [here](https://github.com/PaulPauls/Tensorflow-Neuroevolution)). The examples employ multiple neuroevolution algorithms, problem environments and approaches to the problem and will be steadily extended in future TFNE releases. The basic approach to solving a problem in TFNE is as follows:

The first step is to decide on which NE algorithm to use and to create a complete configuration for all sections and options of the chosen algorithm. Consult the [documentation](https://tfne.readthedocs.io/) for this step. In this quick example we are choosing the CoDeepNEAT algorithm.

The next step is the instantiation of the problem environment the genomes of the chosen neuroevolution algorithm should be evaluated on. Depending on if the chosen NE algorithm trains the weights of the TF models represented by the algorithm's genomes before assigning them a fitness score, instantiate the problem environment as either weight training or not. In this example we are choosing the CIFAR10 environment. The code so far looks like this:

``` python
    import tfne

    config = tfne.parse_configuration('./config-file.cfg')
    ne_algorithm = tfne.algorithms.CoDeepNEAT(config)
    environment = tfne.environments.CIFAR10nvironment(weight_training=True,
                                                     config=config,
                                                     verbosity=0)
```

The instantiated NE algorithm and evaluation environment are then handed over to the driving force of the evolutionary process - the EvolutionEngine. The EvolutionEngine prepares the evolutionary process and takes care of housekeeping tasks. The parameters max_gnerations and max_fitness specify abort conditions for evolution.

``` python
    engine = tfne.EvolutionEngine(ne_algorithm=ne_algorithm,
                                 environment=environment,
                                 backup_dir_path='./',
                                 max_generations=64,
                                 max_fitness=100.0)
```

The evolutionary process however is not started until calling the train() function of the just set up evolutionary engine. This function returns the best genome - and therefore the best TF model - as judged by the evaluation environment that the evolutionary process could produce.

``` python
    best_genome = engine.train()

    print("Best genome returned by evolution:\n")
    print(best_genome)
```

The evolutionary process evaluates, trains, selects, mutates and speciates the population in a potentially boundless loop that is illustrated in basic architecture of TFNE below. All aspects of TFNE can be customized and adapted to the own requirements as long as they conform with this basic flow of operation. For a more detailed introduction into the usage of TFNE and its configuration options, please consult the [documentation](https://tfne.readthedocs.io/).

<p align="center">
  <img src="./documentation/source/illustrations/tfne_v0.21_entity_sequence_diagram.svg" width="80%" alt="TFNE Architecture"/>
  <em><br>Entity-Sequence Diagram Illustrating the Architecture of TFNE</em>
</p>


--------------------------------------------------------------------------------

### Visualizer ###

The TFNE Visualizer is included in the PyPI package of TFNE and offers visualization of the neuroevolution process for all pre-implemented TFNE algorithms. The TFNE Visualizer can be started as a separate script by executing ``tfne_visualizer`` from the command line or by initializing it via TFNE function call. The illustration below showcases the TFNE Visualizer for CoDeepNEAT population backup.

<p align="center">
  <img src="./documentation/source/illustrations/tfnev_demonstration.gif" width="80%" alt="TFNEV Illustration"/>
  <em><br>TFNE Visualizer for CoDeepNEAT Population Backup</em>
</p>


--------------------------------------------------------------------------------

### Documentation ###

TFNE, the framework, all pre-implemented algorithms as well as all available problem environments are extensively documented. This documentation is available both [online on ReadTheDocs](https://tfne.readthedocs.io/) [![Documentation Status](https://readthedocs.org/projects/tfne/badge/?version=latest)](https://tfne.readthedocs.io/en/latest/?badge=latest) as well as offline in the directory ``documentation/build/html/index.html``.


--------------------------------------------------------------------------------

### Roadmap to *stable* ####

If you are reading this notice, you are looking at the public development version of TFNE. The project is under constant development as we aim to update NEAT, implement additional NE algorithms like DeepNEAT, add a variety of different problem environments, introduce novel research algorithms like SubGraphNEAT and SubGraphDeepNEAT and of course also bug-fix the framework. Each of these efforts is done in a development branch of the [TFNE Github repository](https://github.com/PaulPauls/Tensorflow-Neuroevolution). When these efforts are accomplished and TFNE is thoroughly tested do we hope to release v1.0.

Until then do we appreciate any user of TFNE. Any contribution, beginning with an interesting project, constructice feedback, pointing out bugs or even code contributions is also welcome and we thank you for it.

If you have any feedback, bug-fix or remark, please contact [tfne@paulpauls.de](mailto:tfne@paulpauls.de)


--------------------------------------------------------------------------------

### About ###

Project developed by [Paul Pauls](https://github.com/PaulPauls) in collaboration with [Rezsa Farahani](https://www.linkedin.com/in/rezsa). \
We would like to thank Google and the larger Tensorflow team, who have supported this project since the Google Summer of Code 2019!

