.. |br| raw:: html

   <br />


CoDeepNEAT Overview
===================

CoDeepNEAT Publication
-----------------------

+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Title     | Evolving Deep Neural Networks                                                                                                                                                                              |
+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Authors   | Risto Miikkulainen and Jason Liang and Elliot Meyerson and Aditya Rawal and Dan Fink |br| and Olivier Francon and Bala Raju and Hormoz Shahrzad and Arshak Navruzyan |br| and Nigel Duffy and Babak Hodjat |
+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Booktitle | Artificial Intelligence in the Age of Neural Networks and Brain Computing                                                                                                                                  |
+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Editor    | Robert Kozma and Cesare Alippi and Yoonsuck Choe and Francesco Carlo Morabito                                                                                                                              |
+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Publisher | Amsterdam: Elsevier                                                                                                                                                                                        |
+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
| Year      | 2018                                                                                                                                                                                                       |
+-----------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

* `Link to Journal Publication <https://doi.org/10.1016/B978-0-12-815480-9.00015-3>`_
* `Link to paper in arxiv repository <https://arxiv.org/abs/1703.00548>`_
* `Link to paper in repository of University of Texas, Austin <http://nn.cs.utexas.edu/?miikkulainen:chapter18>`_


--------------------------------------------------------------------------------

CoDeepNEAT Overview
-------------------

The Coevolution Deep NeuroEvolution of Augmemting Topologies (CoDeepNEAT) method was first introduced in 2018 by a Team of researchers at the University of Texas, Austin and Sentient Technologies. CoDeepNEAT is a layer evolving algorithm that aims to exploit repetitive structure in the problem domain. It does so by splitting the genome into two elements that can be combined to form a genome, though which each have their own population that is evolved separately according to the methods of neuroevolution. The two elements that make up a genome are termed the blueprint and the modules. A module is a small deep neural network of predefined complexity and configuration that is intended to represent one instance of the repetitive structure of the problem domain. Those modules are concatenated in an overarching graph, that is the blueprint. This blueprint graph is very similar to the genome graph employed by DeepNEAT though instead of nodes representing layers do the nodes of the blueprint graph represent modules, whereas a single module is often repeated multiple times in a blueprint graph. This neuroevolution method therefore aims to evolve neural network with repetitive topologies to effectively exploit repetitive structure in the problem domain. In its original publication the algorithm has very successfully been applied to image recognition, language modeling and image captioning tasks, even producing new state of the art results if given sufficient time to evolve.


--------------------------------------------------------------------------------

Projects involving CoDeepNEAT
-----------------------------

**Cognizant LEAF framework** - The evolutionary AI framework offered by cognizant. The framework is based on CoDeepNEAT according to the paper introducing LEAF to the public. Cognizant also offers an array of additional resources about evolutionary algorithms and their benefits for business and research.

.. _leafpaper: https://arxiv.org/abs/1902.06827
.. |leafpaper| replace:: **LEAF Paper**
.. _gobeyond: https://www.cognizant.com/aicom/documents/cognizant-sentient-leaf-offering-overview.pdf
.. |gobeyond| replace:: **Evolutionary AI: Go Beyond Prediction with LEAF**
.. _businessdecisions: https://www.cognizant.com/ai/blog/informing-business-decision
.. |businessdecisions| replace:: **How Evolutionary AI will inform future business decisions**
.. _bigthing: https://www.youtube.com/watch?v=lfVvl3E1itc
.. |bigthing| replace:: **Evolutionary AI: The Next Big Thing After Deep Learning**
.. _creativeai: https://www.youtube.com/watch?v=GtkZ-_zgoEg
.. |creativeai| replace:: **Evolutionary Computation Enables Truly Creative AI**

* |leafpaper|_
* |gobeyond|_
* |businessdecisions|_
* |bigthing|_
* |creativeai|_


**[More Projects to be Added in the Future]**

