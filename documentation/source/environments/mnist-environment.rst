MNIST Environment
=================

Overview
--------

"The MNIST database (Modified National Institute of Standards and Technology database) is a large database of handwritten digits that is commonly used for training various image processing systems. The database is also widely used for training and testing in the field of machine learning. It was created by "re-mixing" the samples from NIST's original datasets. The creators felt that since NIST's training dataset was taken from American Census Bureau employees, while the testing dataset was taken from American high school students, it was not well-suited for machine learning experiments. Furthermore, the black and white images from NIST were normalized to fit into a 28x28 pixel bounding box and anti-aliased, which introduced grayscale levels."

    -- *from* |hlink|_

.. _hlink: https://en.wikipedia.org/wiki/MNIST_database

.. |hlink| replace:: *Wikipedia*


--------------------------------------------------------------------------------

Specifications
--------------

+-------------------------------------------------------+----------------------+
| Supports Weight-Training Eval                         |                 True |
+-------------------------------------------------------+----------------------+
| Supports Non-Weight-Training Eval                     |                 True |
+-------------------------------------------------------+----------------------+
| Input Shape                                           |          (28, 28, 1) |
+-------------------------------------------------------+----------------------+
| required Output Shape                                 |                (10,) |
+-------------------------------------------------------+----------------------+

The fitness is calculated through the keras accuracy metric, calculating the percentage of how many of the test images are classified correctly.


--------------------------------------------------------------------------------

[EVALUATION] Config Parameters
------------------------------

`Weight-Training Evaluation`
""""""""""""""""""""""""""""

``epochs``
  **Value Range**: int > 0

  **Description**: Specifies the amount of epochs a genome phenotype is to be trained on the environment before evaluating its fitness.


``batch_size``
  **Value Range**: int > 0 | None

  **Description**: Supplied batch_size value for the model.fit function. batch_size is the number of training examples used for a single iteration of backpropagating the gradient of the loss function.


`Non-Weight-Training Evaluation`
""""""""""""""""""""""""""""""""

    **None**

