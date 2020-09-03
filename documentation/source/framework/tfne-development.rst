TFNE Development
================

Version History
---------------

v0.21.0 (26th Aug, 2020)
""""""""""""""""""""""""

* Bug fix and performance optimize beta version of CoDeepNEAT, releasing CoDeepNEAT fully with all intended speciation methods
* Implement TFNE Visualizer with capability to process CoDeepNEAT population backups
* Create ReadTheDocs documentation for TFNE
* Implement extensive unit tests and coverage reporting
* Make TFNE available via PyPI


v0.2 (5th June, 2020)
"""""""""""""""""""""

* Extensively rework framework architecture to enable higher modularity, performance and maintainability
* Release beta version of CoDeepNEAT, including basic speciation
* Implement MNIST and CIFAR10 environments with corresponding examples
* Implement full periodic state backup and deserialization


v0.1 (23rd March, 2020)
"""""""""""""""""""""""

* Initial release providing prototyping framework for neuroevolution with TF 2.x
* Implement NEAT algorithm with direct encoding and fully TF translateable phenotype
* Implement OpenAI gym test environments and provide examples in combination with NEAT
* Implement configurable reporting agents that backup the population and visualize genomes each generation


--------------------------------------------------------------------------------

Future ToDos
------------

General
"""""""
* Implement custom tf.keras.preprocessing functions as used for example in CDN paper, using tf.image API/functions. Integrate those preprocessing options into the TFNE environments.
* Implement a test-wise multi-objective evolution environment [e.g. decreasing fitness based on age; decreasing fitness based on genome complexity]

Code Improvements
"""""""""""""""""
* Implement bidirection connections for the phenotype graphs of all NEAT-like algorithms, as they currently only support feedforward graphs.
* Replace numpy with Google JAX backend and test if it improves performance. Especially relevant for vanilla NEAT, which utilizes numpy extensively.
* Attempt to accelerate TFNE and its algorithms (particularly vanilla NEAT)using tf.function

NEAT
""""
* Completely rework NEAT implementation. Accelerate implementation by parallelizing it. Test possible acceleration by using Google JAX.
* Implement flag to either disable bias, to allow bias being a separate node or to integrate bias into nodes

DeepNEAT
""""""""
* Implement DeepNEAT. Consider layers: Attention, AveragePoolingX, BatchNormalization, BiDirectional, ConvX, Dense, Dropout, Embedding, Flatten, GRU, GlobalAvgPoolX, GlobalMaxPoolX, LSTM, MaxPoolingX, SimpleRNN

CoDeepNEAT
""""""""""
* CoDeepNEAT: Create function to register new modules created outside of code structure. This way it is easier to add new modules if you use TFNE exclusively as a library.

SubGraphNEAT
""""""""""""

SubGraphDeepNEAT
""""""""""""""""

Research Ideas
""""""""""""""
* Find way to determine if 2 genomes represent identical phenotypes or identical functions but have different genotypes and are therefore classified as different species, therefore decreasing effective pop size.
* Implement Lamarckian Evolution for weight-training based NE algorithms (DeepNEAT, CoDeepNEAT, etc)
* Use a gram matrix to cluster member vectors and direct evolution

