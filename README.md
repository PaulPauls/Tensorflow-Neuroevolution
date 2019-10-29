## Neuroevolution Framework for Tensorflow 2.0 ##

**Version _beta_**

**PROJECT IN BETA STAGE!**

The Tensorflow-Neuroevolution framework aims to provide a fast prototyping framework for neuroevolution algorithms realized with Tensorflow 2.0. The current core design focuses on understandability, maintainability and extendability to allow for a seamless interchanging of the three main concerns of neuroevolution - the genome encoding, the neuroevolution algorithm and the evaluation environment. The neuroevolution process is driven forward by a central evolution engine, taking care of arising housework tasks of the neuroevolution process, interacting correctly with the population (the abstract collection of all genomes) and executing user supplied additional tasks such as regular backups.
The framework makes heavy use of Tensorflow and its parallelization/speed-up capabilities in the creation of the genome phenotypes (here: the Tensorflow models), their evaluation and their optimization/evolution, while aiming to stay compatible with the rest of the Tensorflow infrastructure as much as possible.

Important benchmark algorithms, encodings and environments have already been implemented and a multitude of more pre-implemented algorithms, genomes and environments are planned, though not yet realized due to the early stage in development.

#### Neuroevolution Algorithms ####

* [X] Neuroevolution of Augmenting Topologies (NEAT), see additional documentation in [`./documentation/algorithms_neat.md`](https://github.com/PaulPauls/Tensorflow-Neuroevolution/blob/master/documentation/algorithm_neat.md)
* [ ] HyperNEAT, see [here](http://axon.cs.byu.edu/~dan/778/papers/NeuroEvolution/stanley3**.pdf)
* [ ] ES-HyperNEAT, see [here](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.365.4332)
* [ ] CoDeepNEAT, see [here](https://arxiv.org/abs/1703.00548)
* [ ] EvoCNN, see [here](https://arxiv.org/abs/1710.10741)
* [ ] Regularized Evolution for Image Classifier, see [here](https://arxiv.org/abs/1802.01548)


#### Genome Encodings ####

* [X] Direct Encoding, see additional documentation in [`./documentation/encoding_direct.md`](https://github.com/PaulPauls/Tensorflow-Neuroevolution/blob/master/documentation/encoding_direct.md)
* [ ] Indirect Encoding
* [ ] Keras Encoding


#### Test Environments ####

* [X] XOR Problem, see additional documentation in [`./documentation/environment_xor.md`](https://github.com/PaulPauls/Tensorflow-Neuroevolution/blob/master/documentation/environment_xor.md)
* [X] OpenAI Gym CartPole, see [here](http://gym.openai.com/envs/CartPole-v1/)
* [ ] Fashion-MNIST, see [here](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/)
* [ ] Digit-MNIST, see [here](http://yann.lecun.com/exdb/mnist/)
* [ ] \[Other OpenAI Gym Environments\]



--------------------------------------------------------------------------------

#### Installation ####

Installation of the system package `graphviz` is required for visualization of the genome phenotypes.



--------------------------------------------------------------------------------

#### Example Usage ####

Example usage demonstrated in folder `./examples/`. Currently the following examples are present and functional:

* `./examples/xor_neat_example/`: Minimal example showing the basic aspects of TFNE by applying the NEAT algorithm to evolve direct-encoded genomes in order to solve the XOR-problem environment.



--------------------------------------------------------------------------------

#### Architecture Documentation ####

Illustration of the architecture showing the entity relations between the core modules and their respective interactions in the sequence diagram:

![Architecture Illustration](https://raw.githubusercontent.com/PaulPauls/Tensorflow-Neuroevolution/master/documentation/illustration_entity_sequence_diagram_tfne.png)



--------------------------------------------------------------------------------

#### Issues ####

see Github _Issues_ tracker: [here](https://github.com/PaulPauls/Tensorflow-Neuroevolution/issues)



--------------------------------------------------------------------------------

#### ToDo Collection ####

* [ ] Differentiate between neat-original and neat-variant by creating different algorithm classes.
    * To create neat-original: only use linear activations; fix node biases to 0; exclude nodes from the weight difference distance calculation; remove species elitism
    * To create neat-variant:
        * parameterize as many aspects of the current neat as possible
        * Seperate 'gene-id' into (topology-id, weights-id) to better differentiate between genomes with the same or different topologies
        * Introduce a parameter allowing to set the the percentage of weights to be mutated in the mutation-weights function, in order to allow a more fine-grained/incremental evolution of weights

* [ ] Reimplement YANA, renaming it 'TopologyEvolvingWeightTraining' (TEWT) algorithm, by adjusting NEAT to evolve/train weights via Tensorflow backpropagation.

* [ ] Reimplement Weight Training option to the XOR environment, differentiating between the two environments' eval funtions via constructor parameter. Earlier implementation discarded due to architecture overhaul.

* [ ] Implement more OpenAI Gym environments (see [here](https://github.com/openai/gym/wiki/Leaderboard))

* [ ] Implement more extensive tests and integrate them with the checks-API.

* [ ] Implement various environments: SuperMario and Sonic OpenAIRetro env, Fashion MNIST env, generalised TF dataset env, Mujoco env.

* [ ] Reimplement Keras layer encoding. Earlier implementation discarded due to architecture overhaul.

* [ ] Performance benchmark the framework, identify and optimize bottlenecks.

* [ ] Minor ToDos:
    * Add documentation for direct encoding



--------------------------------------------------------------------------------

#### Version History ####

> 20. Oct 2019 - Version _beta_
> * Implement significantly sped up non-trainable DirectEncodingModel, achieving significant sped up but not being compatible with the rest of Tensorflows infrastructure
> * Reimplement CartPole Environment
> * Reimplement shallow serialization of population
> * Reimplement GenomeRender, PopulationBackup and Speciation reporting agents
> * Implement various tests and examples, testing the new functionality and environment

> 09. Oct 2019 - Version _beta_
> * Overhaul DirectEncoding to not only direct encode topology, but also weights in the genes
> * Implement the NE algorithm NEAT
> * Adjust the regular XOR environment to only evaluate, not train, the supplied genomes
> * Overhaul the population class, saving its genomes in hashtables now and accessing them via keys, saving resources
> * Switch the logging method to abseil-py, as recommended for TF 2.0
> * Extensively documenting everything, extensively updating README and introducing new 'documentation' folder

> 22. Aug 2019 - Version _alpha_
> * Fix Bug where direct-encoding model uses default_activation for out_activation
> * Minor refactoring to decrease coupling as well as clarify and optimize code
> * Add early_stop functionality to XOR environment
> * Add extensive inline documentaion
> * Publish ToDo collection, Bugs and Architecture Documentation



--------------------------------------------------------------------------------

#### About ####

Project developed by [Paul Pauls](https://github.com/PaulPauls) in collaboration with [Rezsa Farahani](https://github.com/rezsa)



