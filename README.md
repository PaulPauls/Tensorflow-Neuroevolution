## Neuroevolution Framework for Tensorflow 2.0 ##

**Version _beta_**

**PROJECT IN BETA STAGE!**

The Tensorflow-Neuroevolution project aims to provide a fast prototyping framework for neuroevolution algorithms realized with Tensorflow 2.0. It makes use of Tensorflows MultiAgent environment whenever possible to automatically deploy the Neuroevolution methods to the maximum degree of parallelization.

A multitude of pre-implemented algorithms, genomes and environments are planned, though not yet realized due to the early stage in development.

#### Neuroevolution Algorithms ####

* [X] Neuroevolution of Augmenting Topologies (NEAT), see [here](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
* [ ] HyperNEAT, see [here](http://axon.cs.byu.edu/~dan/778/papers/NeuroEvolution/stanley3**.pdf)
* [ ] ES-HyperNEAT, see [here](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.365.4332)
* [ ] CoDeepNEAT, see [here](https://arxiv.org/abs/1703.00548)
* [ ] EvoCNN, see [here](https://arxiv.org/abs/1710.10741)
* [ ] Regularized Evolution for Image Classifier, see [here](https://arxiv.org/abs/1802.01548)


#### Genome Encodings ####

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

Example usage demonstrated in folder `examples/` and `tests/`. Currently the following examples are present and functional:

* `examples/xor_neat_example/`: Applying the NEAT neuroevolution algorithm to evolve direct-encoded genomes in order to solve the XOR-problem environment



--------------------------------------------------------------------------------

#### Architecture Documentation ####

Illustration of the architecture showing the entity relations between the core modules and their respective interactions in the sequence diagram:

![Architecture Illustration](https://raw.githubusercontent.com/PaulPauls/Tensorflow-Neuroevolution/master/.architecture_illustration/Entity_Sequence_Diagram_TFNE-Framework.png)



--------------------------------------------------------------------------------

#### Issues ####

see Github _Issues_ tracker: [here](https://github.com/PaulPauls/Tensorflow-Neuroevolution/issues)



--------------------------------------------------------------------------------

#### ToDo Collection ####

**TBD**


--------------------------------------------------------------------------------

#### Version History ####

> 09. Okt 2019 - Version _beta_
> * **TBD**

> 22. Aug 2019 - Version _alpha_
> * Fix Bug where direct-encoding model uses default_activation for out_activation
> * Minor refactoring to decrease coupling as well as clarify and optimize code
> * Add early_stop functionality to XOR environment
> * Add extensive inline documentaion
> * Publish ToDo collection, Bugs and Architecture Documentation 

> 19. Aug 2019 - Version _alpha_
> * Refactoring of framework to generalize APIs further. 
> * Implementation of Direct Encoding enabling arbitrary feedforward topologies to be encoded in explicitely defined genotypes
> * Implementation of the XOR-problem environment
> * Implementation of a basic NE algorithm (YANA), mostly intended for Direct Encoding applied to the XOR environment
> * Implementation of the xor_example showcasing the framework
> * Multiple minor bug fixes



--------------------------------------------------------------------------------

#### About ####

Project developed by [Paul Pauls](https://github.com/PaulPauls) in collaboration with [Rezsa Farahani](https://github.com/rezsa)


