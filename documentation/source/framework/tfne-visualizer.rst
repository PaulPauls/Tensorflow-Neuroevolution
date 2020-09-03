TFNE Visualizer
===============

The TFNE Visualizer is included in the PyPI package of TFNE and offers visualization of the neuroevolution process for all pre-implemented TFNE algorithms. The TFNE Visualizer can be started as a separate script by executing ``tfne_visualizer`` from the command line or by initializing it via the following function of TFNE, optionally supplying the path to the tfne state backup that is to be visualized:

.. code-block:: python

    import tfne

    tfne.start_visualizer('./tfne_state_backups/')


If no path to the TFNE state backup is supplied in the function call or the visualizer is started as a script from the command line, does TFNEV show the welcome window, as seen in the figure below. The welcome window allows to open a file dialog and to choose the TFNE state backup via the familiar directory interface.

.. figure:: ../illustrations/tfnev_welcome_illustration.png
   :align: center

   TFNE Visualizer Welcome Window


Once a TFNE state backup folder has been selected does the visualizer automatically load all population backups, analyze the type of backup and open the visualization window for the appropriate NE algorithm. The figure below illustrates the visualization of an example CoDeepNEAT state trace that is provided with the CoDeepNEAT-MNIST example that can be found `here <https://github.com/PaulPauls/Tensorflow-Neuroevolution/tree/master/examples/codeepneat/codeepneat_mnist_example>`_.

.. figure:: ../illustrations/tfnev_demonstration.gif
   :align: center

   TFNE Visualizer CoDeepNEAT Window

