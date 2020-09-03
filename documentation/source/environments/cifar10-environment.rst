CIFAR10 Environment
===================

Overview
--------

"The CIFAR-10 dataset (Canadian Institute For Advanced Research) is a collection of images that are commonly used to train machine learning and computer vision algorithms. It is one of the most widely used datasets for machine learning research. The CIFAR-10 dataset contains 60,000 32x32 color images in 10 different classes. The 10 different classes represent airplanes, cars, birds, cats, deer, dogs, frogs, horses, ships, and trucks. There are 6,000 images of each class.

Computer algorithms for recognizing objects in photos often learn by example. CIFAR-10 is a set of images that can be used to teach a computer how to recognize objects. Since the images in CIFAR-10 are low-resolution (32x32), this dataset can allow researchers to quickly try different algorithms to see what works. Various kinds of convolutional neural networks tend to be the best at recognizing the images in CIFAR-10.

CIFAR-10 is a labeled subset of the 80 million tiny images dataset. When the dataset was created, students were paid to label all of the images."

    -- *from* |hlink|_

.. _hlink: https://en.wikipedia.org/wiki/CIFAR-10

.. |hlink| replace:: *Wikipedia*


--------------------------------------------------------------------------------

Specifications
--------------

+-------------------------------------------------------+----------------------+
| Supports Weight-Training Eval                         |                 True |
+-------------------------------------------------------+----------------------+
| Supports Non-Weight-Training Eval                     |                 True |
+-------------------------------------------------------+----------------------+
| Input Shape                                           |          (32, 32, 3) |
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

