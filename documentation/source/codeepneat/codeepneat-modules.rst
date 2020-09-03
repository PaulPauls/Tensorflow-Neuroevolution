CoDeepNEAT Modules
==================

This section of the CoDeepNEAT documentation covers pre-implemented module classes that are available in TFNE. An important part of CoDeepNEAT is its extendability by defining one's own module classes and evolve them. See the last paragraph of this section below to learn how to add your own modules to TFNE and what to consider by inheriting from the TFNE abstract module interface.


--------------------------------------------------------------------------------

Dense-Dropout Module
--------------------

This is a very simple module. The represented deep neural network consists of a guaranteed dense layer followed by an optional dropout layer. This module has 7 parameters that can be set by the configuration. Of these 7 parameters, 5 are providing parameters for the encoded deep neural network layer and 2 parameters are relevant for the TFNE CoDeepNEAT evolution. Explicitly:

  * ``merge_method`` - list of TF deserializable strings specifying a valid merge layer
  * ``dropout_flag`` - float [0; 1.0] specifying the probability of the optional dropout layer in the final module DNN

The full list of all parameters can be seen in the paragraph below. Please consult the excellent TF API for the documentation of the parameters concerning the layers.

Downsampling mismatched input, mutating the parameters, module crossover and parameter distance calculation are implemented as follows for this module:

**Downsampling** - Downsampling not yet implemented for this module.

**Mutation** - Continuous parameters are mutated by drawing a random value from a normal distribution with the parent parameter value as the *mean* and the *stddev* set via the configuration. Categorical parameters supplied via a list of preselected values are mutated by choosing a new value randomly.

**Crossover** - Modules are crossed over by averaging out their continuous parameters and choosing the value of the fitter module for categorical parameters.

**Parameter Distance** - First determining the congruence between 2 module parameter sets and then subtract the actual congruence from the perfect congruence of 1.0. The congruence of continuous parameters is calculated by their relative distance. The congruence of categorical parameters is either 1.0 in case they are the same or it's 1 divided by the amount of possible values for that specific parameter.


[MODULE_DENSEDROPOUT] Config Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cfg

    [MODULE_DENSEDROPOUT]
    merge_method = ...
    units        = ...
    activation   = ...
    kernel_init  = ...
    bias_init    = ...
    dropout_flag = ...
    dropout_rate = ...


All layer parameters can be configured in 3 different ways:

If the layer parameter should have a fixed value, specify that fixed value in the configuration.

If the layer parameter is continuous and the optimal value is to be determined by evolution, specify the minimum (``min``) and maximum (``max``) value of the layer parameter as well as the possible step (``step``) between different values in a python dict style. Also specify the standard deviation (``stddev``) that should be applied when mutating the parameter and a new value is chosen for the parameter from a normal distribution.

If the layer parameter is discrete or should only be chosen from a preselected list and the optimal value is to be determined by evolution, specify all possible values of the parameter in a python list style.


--------------------------------------------------------------------------------

Conv2D-MaxPool2D-Dropout Module
-------------------------------

The deep neural network represented by this module consists of a guaranteed Conv2D layer, followed by an optional MaxPooling2D layer, followed by an optional Dropout layer. This module has 12 parameters that can be set by the configuration. Of these 12 parameters, 9 are providing parameters for the encoded deep neural network layer and 3 parameters are relevant for the TFNE CoDeepNEAT evolution. Explicitly:

  * ``merge_method`` - list of TF deserializable strings specifying a valid merge layer
  * ``max_pool_flag`` - float [0; 1.0] specifying the probability of the optional Max Pooling layer in the final module DNN
  * ``dropout_flag`` - float [0; 1.0] specifying the probability of the optional dropout layer in the final module DNN

The full list of all parameters can be seen in the paragraph below. Please consult the excellent TF API for the documentation of the parameters concerning the layers.

Downsampling mismatched input, mutating the parameters, module crossover and parameter distance calculation are implemented as follows for this module:

**Downsampling** - Mismatched input is downsampled by inserting an additional Conv2D layer between the mismatched input and the module DNN. This Conv2D layer converts the input to the largest possible input shape to preserve as much information as possible.

**Mutation** - Continuous parameters are mutated by drawing a random value from a normal distribution with the parent parameter value as the *mean* and the *stddev* set via the configuration. Categorical parameters supplied via a list of preselected values are mutated by choosing a new value randomly.

**Crossover** - Modules are crossed over by averaging out their continuous parameters and choosing the value of the fitter module for categorical parameters.

**Parameter Distance** - First determining the congruence between 2 module parameter sets and then subtract the actual congruence from the perfect congruence of 1.0. The congruence of continuous parameters is calculated by their relative distance. The congruence of categorical parameters is either 1.0 in case they are the same or it's 1 divided by the amount of possible values for that specific parameter.


[MODULE_CONV2DMAXPOOL2DDROPOUT] Config Parameters
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: cfg

   [MODULE_CONV2DMAXPOOL2DDROPOUT]
   merge_method  = ...
   filters       = ...
   kernel_size   = ...
   strides       = ...
   padding       = ...
   activation    = ...
   kernel_init   = ...
   bias_init     = ...
   max_pool_flag = ...
   max_pool_size = ...
   dropout_flag  = ...
   dropout_rate  = ...


All layer parameters can be configured in 3 different ways:

If the layer parameter should have a fixed value, specify that fixed value in the configuration.

If the layer parameter is continuous and the optimal value is to be determined by evolution, specify the minimum (``min``) and maximum (``max``) value of the layer parameter as well as the possible step (``step``) between different values in a python dict style. Also specify the standard deviation (``stddev``) that should be applied when mutating the parameter and a new value is chosen for the parameter from a normal distribution.

If the layer parameter is discrete or should only be chosen from a preselected list and the optimal value is to be determined by evolution, specify all possible values of the parameter in a python list style.


--------------------------------------------------------------------------------

Defining Your Own Modules
-------------------------

Defining your own TFNE CoDeepNEAT modules is simple as all required functionality is dictated by the abstract module interface seen below. The required functionality and output of each method is documented in its respective docstrings. This abstract module interface can be included by inheriting from ``tfne.encodings.codeepneat.modules.CoDeepNEATModuleBase``.

To make the newly created TFNE compatible module usable by TFNE you have to include an association between the module string name and the module class implementation in the file ``tfne/encodings/codeepneat/modules/codeepneat_module_association.py``. We are working on making this process simpler in a future release.


CoDeepNEAT Abstract Module Interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    class CoDeepNEATModuleBase(object, metaclass=ABCMeta):
        """
        Base class and interface for TFNE CoDeepNEAT compatible modules, ensuring that modules provide layer creation,
        downsampling, mutation and crossover functionality. This base class also provides common functionality required
        by all modules like parameter saving and simple setter/getter methods.
        """

        def __init__(self, config_params, module_id, parent_mutation, dtype):
            """
            Base class of all TFNE CoDeepNEAT modules, saving common parameters.
            @param config_params: dict of the module parameter range supplied via config
            @param module_id: int of unique module ID
            @param parent_mutation: dict summarizing the mutation of the parent module
            @param dtype: string of deserializable TF dtype
            """
            self.config_params = config_params
                self.module_id = module_id
            self.parent_mutation = parent_mutation
            self.dtype = dtype
            self.fitness = 0

        @abstractmethod
        def __str__(self) -> str:
            """
            @return: string representation of the module
            """
            raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement '__str__()'")

        @abstractmethod
        def create_module_layers(self) -> (tf.keras.layers.Layer, ...):
            """
            Instantiate all TF layers represented by the module and return as iterable tuple
            @return: iterable tuple of instantiated TF layers
            """
            raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'create_module_layers()'")

        @abstractmethod
        def create_downsampling_layer(self, in_shape, out_shape) -> tf.keras.layers.Layer:
            """
            Create layer associated with this module that downsamples the non compatible input shape to the input shape of
            the current module, which is the output shape of the downsampling layer.
            @param in_shape: int tuple of incompatible input shape
            @param out_shape: int tuple of the intended output shape of the downsampling layer
            @return: instantiated TF keras layer that can downsample incompatible input shape to a compatible input shape
            """
            raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'create_downsampling_layer()'")

        @abstractmethod
        def create_mutation(self,
                            offspring_id,
                            max_degree_of_mutation) -> CoDeepNEATModuleBase:
            """
            Create a mutated module and return it
            @param offspring_id: int of unique module ID of the offspring
            @param max_degree_of_mutation: float between 0 and 1 specifying the maximum degree of mutation
            @return: instantiated TFNE CoDeepNEAT module with mutated parameters
            """
            raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'create_mutation()'")

        @abstractmethod
        def create_crossover(self,
                             offspring_id,
                             less_fit_module,
                             max_degree_of_mutation) -> CoDeepNEATModuleBase:
            """
            Create a crossed over module and return it
            @param offspring_id: int of unique module ID of the offspring
            @param less_fit_module: second module of same type with less fitness
            @param max_degree_of_mutation: float between 0 and 1 specifying the maximum degree of mutation
            @return: instantiated TFNE CoDeepNEAT module with crossed over parameters
            """
            raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'create_crossover()'")

        @abstractmethod
        def serialize(self) -> dict:
            """
            @return: serialized constructor variables of the module as json compatible dict
            """
            raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'serialize()'")

        @abstractmethod
        def get_distance(self, other_module) -> float:
            """
            Calculate the distance between 2 TFNE CoDeepNEAT modules with high values indicating difference, low values
            indicating similarity
            @param other_module: second TFNE CoDeepNEAT module to which the distance has to be calculated
            @return: float between 0 and 1. High values indicating difference, low values indicating similarity
            """
            raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'get_distance()'")

        @abstractmethod
        def get_module_type(self) -> str:
            """
            @return: string representation of module type as used in CoDeepNEAT config
            """
            raise NotImplementedError("Subclass of CoDeepNEATModuleBase does not implement 'get_module_name()'")

        def set_fitness(self, fitness):
            self.fitness = fitness

        def get_id(self) -> int:
            return self.module_id

        def get_fitness(self) -> float:
            return self.fitness

        def get_merge_method(self) -> dict:
            return self.merge_method

