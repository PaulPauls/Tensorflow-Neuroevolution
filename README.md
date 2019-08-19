## Neuroevolution Framework for Tensorflow 2.0 ##

**Version _alpha_**

**PROJECT IN EARLY ALPHA STAGE!**

The Tensorflow-Neuroevolution project aims to provide a fast prototyping framework for neuroevolution algorithms realized with Tensorflow 2.0. It makes use of Tensorflows MultiAgent environment whenever possible to automatically deploy the Neuroevolution methods to the maximum degree of parallelization.

A multitude of pre-implemented algorithms, genomes and environments are planned, though not yet realized due to the early stage in development.

#### Neuroevolution Algorithms ####

* [ ] Neuroevolution of Augmenting Topologies (NEAT), see [here](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
* [ ] HyperNEAT, see [here](http://axon.cs.byu.edu/~dan/778/papers/NeuroEvolution/stanley3**.pdf)
* [ ] ES-HyperNEAT, see [here](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.365.4332)
* [ ] CoDeepNEAT, see [here](https://arxiv.org/abs/1703.00548)
* [ ] EvoCNN, see [here](https://arxiv.org/abs/1710.10741)
* [ ] Regularized Evolution for Image Classifier, see [here](https://arxiv.org/abs/1802.01548)


#### Genome Encodings ####

* [ ] Binary Encoding
* [X] Direct Encoding
* [ ] Indirect Encoding 
* [ ] Keras Encoding


#### Test Environments ####

* [X] XOR Problem
* [ ] OpenAI Gym CartPole, see [here](http://gym.openai.com/envs/CartPole-v1/)
* [ ] Fashion-MNIST, see [here](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/)
* [ ] Digit-MNIST, see [here](http://yann.lecun.com/exdb/mnist/)
* [ ] \[Other OpenAI Gym Environments\]



--------------------------------------------------------------------------------

#### Example Usage ####

Example Usage demonstrated in folder `examples/` and `tests/`. Currently the following examples are present and functional:

* `examples/example_yanaAlg_directEnc_xorEnv`: Applying the rudimentary YANA neuroevolution algorithm to evolve direct-encoded genomes in order to solve the XOR-problem environment 



--------------------------------------------------------------------------------

#### Architecture Documentation ####

ToDo



--------------------------------------------------------------------------------

#### Version History ####

> 19. Aug 2019 - Version _alpha_
> * Refactoring of framework to generalize APIs further. 
> * Implementation of Direct Encoding enabling arbitrary feedforward topologies to be encoded in explicitely defined genotypes
> * Implementation of the XOR-problem environment
> * Implementation of a basic NE algorithm (YANA), mostly intended for Direct Encoding applied to the XOR environment
> * Implementation of the xor_example showcasing the framework
> * Multiple minor bug fixes

The development diary for this project can be found [here](https://paulpauls.github.io/Tensorflow-Neuroevolution/).

--------------------------------------------------------------------------------

#### Issues ####

ToDo



--------------------------------------------------------------------------------

#### About ####

Project developed by Paul Pauls in collaboration with Rezsa Farahani in the context of Tensorflow's Google Summer of Code Program in 2019.

