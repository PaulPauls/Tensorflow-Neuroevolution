## The Tensorflow Neuroevolution Framework ##

**Version 0.2 BETA**

The **TensorFlow NeuroEvolution [abbr. TFNE]** framework aims to provide a fast prototyping framework for neuroevolution algorithms realized with Tensorflow 2.x. While the framework is certainly optimized for high performance does the design focus on maintainability, modularity and extendability by allowing for a seamless interchange of the three main concerns of neuroevolution - the genome encoding, the neuroevolution algorithm and the evaluation environment.
TFNE comes with a variety of pre-implemented neuroevolution algorithms, encodings and environments that are all highly optimized and extensively documented. The phenotypes (Neural Networks encoded through the genome) created through the pre-implemented genome encodings make heavy use of Tensorflow and its internal optimization, as the Tensorflow models are exclusively created through the keras functional API and therefore allow for high-performance as well as full compatibility to the rest of the Tensorflow ecosystem. This eliminates a common dread in other evolutionary frameworks in which the evolved genome and its phenotype is not compatible with other machine learning frameworks.
All neuroevolution algorithms are controlled through the central evolution engine, which initializes required libraries and initiates the single steps of the evolutionary process. The evolution engine allows for the usage of custom or prototyped algorithms, encodings or environments, given they adhere to the abstract interfaces supplied for each neuroevolution aspect.

**Notes on BETA version, which will be fixed in v0.2:**
* CoDeepNEAT as of now only supports the basic speciation scheme
* TFNE visualizer tool not yet published
* Extensive readthedocs Documentation not yet created
* Extensive tests not yet implemented


---------------------------------------------------------

#### Pre-Implemented Algorithms, Encodings and Environments ####

*Neuroevolution Algorithms:*
[x] (see v0.1) Neuroevolution of Augmenting Topologies (NEAT) [paper | doc | code]
[ ] HyperNEAT [paper | doc | code]
[ ] ES-HyperNEAT [paper | doc | code]
[ ] DeepNEAT [paper | doc | code]
[X] CoDeepNEAT [paper | doc | code]

*Genome Encodings:*
[X] (see v0.1) NEAT Encoding [doc | code]
[X] CoDeepNEAT Encoding [doc | code]

*Genome Encodings:*
[X] XOR Problem [doc | code]
[X] (see v0.1) OpenAI Gym CartPole [paper | doc | code]
[X] CIFAR10 [paper | doc | code]


---------------------------------------------------------

#### Installation ####

TFNE requires Python 3.7 or later.
For the required Python packages, see [`./requirements.txt`](https://github.com/PaulPauls/Tensorflow-Neuroevolution/blob/master/requirements.txt)
Installation of the system package `graphviz` is required for the usage of the TFNE visualizer.


---------------------------------------------------------

#### Documentation ####

Online documentation: https://tfne.readthedocs.io
Offline documentation: [`./documentation/build/html/index.html`](https://github.com/PaulPauls/Tensorflow-Neuroevolution/blob/master/documentation/build/html/index.html)
Both documentations are identical.


---------------------------------------------------------

#### Examples ####

For a variety of examples using different algorithms, encodings and environments, see: [`./examples/`](https://github.com/PaulPauls/Tensorflow-Neuroevolution/blob/master/examples/)
Example usage of TFNE in a video format will be recorded, uploaded and linked here in the coming months.


---------------------------------------------------------

#### About ####

Project developed by [Paul Pauls](https://github.com/PaulPauls) in collaboration with [Rezsa Farahani](https://www.linkedin.com/in/rezsa).
We would like to thank Google as well as the Tensorflow Team, who have supported this project during the Google Summer of Code 2019 and 2020.

