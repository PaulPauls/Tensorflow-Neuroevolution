## Neuroevolution Framework for Tensorflow 2.0 ##


The Tensorflow-Neuroevolution project aims to provide a fast prototyping framework for neuroevolution algorithms realized with Tensorflow 2.0. It makes use of Tensorflows MultiAgent environment whenever possible to automatically deploy the Neuroevolution methods to the maximum degree of parallelization.

**PROJECT IN EARLY ALPHA STAGE!**

The development diary for this project can be found [here](https://paulpauls.github.io/Tensorflow-Neuroevolution/).


------------------------------------------------------------------------

A multitude of pre-implemented algorithms, genomes and environments are planned, though none are yet realized due to the early stage in development.

#### Neuroevolution Algorithms ####

* [ ] Neuroevolution of Augmenting Topologies (NEAT), see [here](http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf)
* [ ] HyperNEAT, see [here](http://axon.cs.byu.edu/~dan/778/papers/NeuroEvolution/stanley3**.pdf)
* [ ] ES-HyperNEAT, see [here](http://citeseerx.ist.psu.edu/viewdoc/summary?doi=10.1.1.365.4332)
* [ ] CoDeepNEAT, see [here](https://arxiv.org/abs/1703.00548)
* [ ] EvoCNN, see [here](https://arxiv.org/abs/1710.10741)
* [ ] Regularized Evolution for Image Classifier, see [here](https://arxiv.org/abs/1802.01548)


#### Genome Encodings ####

* [ ] Binary Encoding
* [ ] Direct Encoding
* [ ] Indirect Encoding 
* [x] Keras Encoding


#### Test Environments ####

* [x] Fashion-MNIST, see [here](https://research.zalando.com/welcome/mission/research-projects/fashion-mnist/)
* [ ] Digit-MNIST, see [here](http://yann.lecun.com/exdb/mnist/)
* [ ] OpenAI Gym CartPole, see [here](http://gym.openai.com/envs/CartPole-v1/)
* [ ] \[Other OpenAI Gym Environments\]


------------------------------------------------------------------------

#### Examples ####

Currently only a single development example available (fashion_mnist_example.py) that utilizes the test-algorithm 'YANA' to develop a Keras Encoding in order to solve the Fashion-MNIST environment. The 'YANA' algorithm has no basis in research but serves only to test the complete framework. 


------------------------------------------------------------------------

#### Issues ####

Issues with the framework are common at the moment and are tracked in the
[development diary](https://paulpauls.github.io/Tensorflow-Neuroevolution/) for now.


------------------------------------------------------------------------

#### About ####

Project developed by Paul Pauls under guidance of Rezsa Farahani in the context of the Google Summer of Code Program 2019.
