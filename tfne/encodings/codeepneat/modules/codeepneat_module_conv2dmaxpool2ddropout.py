from __future__ import annotations

import math
import random

import numpy as np
import tensorflow as tf

from .codeepneat_module_base import CoDeepNEATModuleBase
from ....helper_functions import round_with_step


class CoDeepNEATModuleConv2DMaxPool2DDropout(CoDeepNEATModuleBase):
    """"""

    def __init__(self,
                 config_params,
                 module_id,
                 parent_mutation,
                 merge_method,
                 filters,
                 kernel_size,
                 strides,
                 padding,
                 activation,
                 kernel_init,
                 bias_init,
                 max_pool_flag,
                 max_pool_size,
                 dropout_flag,
                 dropout_rate,
                 self_initialize=False):
        """"""
        # Register the dict listing the module parameter range specified in the config
        self.config_params = config_params

        # Register the module parameters
        super().__init__(module_id, parent_mutation)
        self.merge_method = merge_method
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.max_pool_flag = max_pool_flag
        self.max_pool_size = max_pool_size
        self.dropout_flag = dropout_flag
        self.dropout_rate = dropout_rate

        # If self initialization flag is provided, initialize the module parameters as they are currently set to None
        if self_initialize:
            self.initialize()

    def __str__(self) -> str:
        """"""
        return "CoDeepNEAT Conv2D MaxPool Dropout Module | ID: {:>6} | Fitness: {:>6} | Filters: {:>4} | " \
               "Kernel: {:>6} | Activ: {:>6} | Pool Size: {:>6} | Dropout: {:>4}" \
            .format('#' + str(self.module_id),
                    self.fitness,
                    self.filters,
                    str(self.kernel_size),
                    self.activation,
                    "None" if self.max_pool_flag is False else str(self.max_pool_size),
                    "None" if self.dropout_flag is False else self.dropout_rate)

    def create_module_layers(self, dtype) -> [tf.keras.layers.Layer, ...]:
        """"""
        # Create iterable that contains all layers concatenated in this module
        module_layers = list()

        # Create the basic keras Convolutional 2D layer, needed in all variants of the module
        conv_layer = tf.keras.layers.Conv2D(filters=self.filters,
                                            kernel_size=self.kernel_size,
                                            strides=self.strides,
                                            padding=self.padding,
                                            activation=self.activation,
                                            kernel_initializer=self.kernel_init,
                                            bias_initializer=self.bias_init,
                                            dtype=dtype)
        module_layers.append(conv_layer)

        # If max pooling flag present, add the max pooling layer as configured to the module layers iterable
        if self.max_pool_flag:
            max_pool_layer = tf.keras.layers.MaxPool2D(pool_size=self.max_pool_size,
                                                       dtype=dtype)
            module_layers.append(max_pool_layer)

        # If dropout flag present, add the dropout layer as configured to the module layers iterable
        if self.dropout_flag:
            dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_rate,
                                                    dtype=dtype)
            module_layers.append(dropout_layer)

        # Return the iterable containing all layers present in the module
        return module_layers

    def create_downsampling_layer(self, in_shape, out_shape, dtype) -> tf.keras.layers.Layer:
        """"""
        # As the Conv2DMaxPool2DDropout module downsamples with a Conv2D layer, assure that the input and output shape
        # are of dimension 4 and that the second and third channel are identical
        if not (len(in_shape) == 4 and len(out_shape) == 4) \
                or in_shape[1] != in_shape[2] \
                or out_shape[1] != out_shape[2]:
            raise NotImplementedError(f"Downsampling Layer for the shapes {in_shape} and {out_shape}, not having 4"
                                      f"channels or differing second and third channels has not yet been implemented "
                                      f"for the Conv2DMaxPool2DDropout module")

        # If Only the second and thid channel have to be downsampled then carry over the size of the fourth channel and
        # adjust the kernel size to result in the adjusted second and third channel size
        if out_shape[1] is not None and out_shape[3] is None:
            filters = in_shape[3]
            kernel_size = in_shape[1] - out_shape[1] + 1
            return tf.keras.layers.Conv2D(filters=filters,
                                          kernel_size=kernel_size,
                                          strides=(1, 1),
                                          padding='valid',
                                          activation=None)

        # If Only the fourth channel has to be downsampled then carry over the size of the second and fourth channel and
        # adjust the filters to result in the adjusted fourth channel size
        elif out_shape[1] is None and out_shape[3] is not None:
            filters = out_shape[3]
            kernel_size = in_shape[1]
            return tf.keras.layers.Conv2D(filters=filters,
                                          kernel_size=kernel_size,
                                          strides=(1, 1),
                                          padding='same',
                                          activation=None)

        # If the second, third and fourth channel have to be downsampled adjust both the filters and kernel size
        # accordingly to result in the desired output shape
        elif out_shape[1] is not None and out_shape[3] is not None:
            filters = out_shape[3]
            kernel_size = in_shape[1] - out_shape[1] + 1
            return tf.keras.layers.Conv2D(filters=filters,
                                          kernel_size=kernel_size,
                                          strides=(1, 1),
                                          padding='valid',
                                          activation=None)
        else:
            raise RuntimeError(f"Downsampling to output shape {out_shape} from input shape {in_shape} not possible"
                               f"with a Conv2D layer")

    def initialize(self):
        """"""
        # Uniformly randomly set module parameters
        self.merge_method = random.choice(self.config_params['merge_method'])
        random_filters = random.randint(self.config_params['filters']['min'],
                                        self.config_params['filters']['max'])
        self.filters = round_with_step(random_filters,
                                       self.config_params['filters']['min'],
                                       self.config_params['filters']['max'],
                                       self.config_params['filters']['step'])
        self.kernel_size = random.choice(self.config_params['kernel_size'])
        self.strides = random.choice(self.config_params['strides'])
        self.padding = random.choice(self.config_params['padding'])
        self.activation = random.choice(self.config_params['activation'])
        self.kernel_init = random.choice(self.config_params['kernel_init'])
        self.bias_init = random.choice(self.config_params['bias_init'])
        self.max_pool_flag = random.random() < self.config_params['max_pool_flag']
        self.max_pool_size = random.choice(self.config_params['max_pool_size'])
        self.dropout_flag = random.random() < self.config_params['dropout_flag']
        random_dropout_rate = random.uniform(self.config_params['dropout_rate']['min'],
                                             self.config_params['dropout_rate']['max'])
        self.dropout_rate = round(round_with_step(random_dropout_rate,
                                                  self.config_params['dropout_rate']['min'],
                                                  self.config_params['dropout_rate']['max'],
                                                  self.config_params['dropout_rate']['step']), 4)

    def create_mutation(self,
                        offspring_id,
                        max_degree_of_mutation) -> (int, CoDeepNEATModuleConv2DMaxPool2DDropout):
        """"""
        # Copy the parameters of this parent module for the parameters of the offspring
        offspring_params = {'merge_method': self.merge_method,
                            'filters': self.filters,
                            'kernel_size': self.kernel_size,
                            'strides': self.strides,
                            'padding': self.padding,
                            'activation': self.activation,
                            'kernel_init': self.kernel_init,
                            'bias_init': self.bias_init,
                            'max_pool_flag': self.max_pool_flag,
                            'max_pool_size': self.max_pool_size,
                            'dropout_flag': self.dropout_flag,
                            'dropout_rate': self.dropout_rate}

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': self.module_id,
                           'mutation': 'mutation',
                           'mutated_params': dict()}

        # Determine exact integer amount of parameters to be mutated, though minimum is 1
        param_mutation_count = math.ceil(max_degree_of_mutation * 12)

        # Uniform randomly choose the parameters to be mutated
        parameters_to_mutate = random.sample(range(12), k=param_mutation_count)

        # Mutate offspring parameters. Categorical parameters are chosen randomly from all available values. Sortable
        # parameters are perturbed through a random normal distribution with the current value as mean and the config
        # specified stddev
        for param_to_mutate in parameters_to_mutate:
            if param_to_mutate == 0:
                offspring_params['merge_method'] = random.choice(self.config_params['merge_method'])
                parent_mutation['mutated_params']['merge_method'] = self.merge_method
            elif param_to_mutate == 1:
                perturbed_filters = int(np.random.normal(loc=self.filters,
                                                         scale=self.config_params['filters']['stddev']))
                offspring_params['filters'] = round_with_step(perturbed_filters,
                                                              self.config_params['filters']['min'],
                                                              self.config_params['filters']['max'],
                                                              self.config_params['filters']['step'])
                parent_mutation['mutated_params']['filters'] = self.filters
            elif param_to_mutate == 2:
                offspring_params['kernel_size'] = random.choice(self.config_params['kernel_size'])
                parent_mutation['mutated_params']['kernel_size'] = self.kernel_size
            elif param_to_mutate == 3:
                offspring_params['strides'] = random.choice(self.config_params['strides'])
                parent_mutation['mutated_params']['strides'] = self.strides
            elif param_to_mutate == 4:
                offspring_params['padding'] = random.choice(self.config_params['padding'])
                parent_mutation['mutated_params']['padding'] = self.padding
            elif param_to_mutate == 5:
                offspring_params['activation'] = random.choice(self.config_params['activation'])
                parent_mutation['mutated_params']['activation'] = self.activation
            elif param_to_mutate == 6:
                offspring_params['kernel_init'] = random.choice(self.config_params['kernel_init'])
                parent_mutation['mutated_params']['kernel_init'] = self.kernel_init
            elif param_to_mutate == 7:
                offspring_params['bias_init'] = random.choice(self.config_params['bias_init'])
                parent_mutation['mutated_params']['bias_init'] = self.bias_init
            elif param_to_mutate == 8:
                offspring_params['max_pool_flag'] = not self.max_pool_flag
                parent_mutation['mutated_params']['max_pool_flag'] = self.max_pool_flag
            elif param_to_mutate == 9:
                offspring_params['max_pool_size'] = random.choice(self.config_params['max_pool_size'])
                parent_mutation['mutated_params']['max_pool_size'] = self.max_pool_size
            elif param_to_mutate == 10:
                offspring_params['dropout_flag'] = not self.dropout_flag
                parent_mutation['mutated_params']['dropout_flag'] = self.dropout_flag
            else:  # param_to_mutate == 11:
                perturbed_dropout_rate = np.random.normal(loc=self.dropout_rate,
                                                          scale=self.config_params['dropout_rate']['stddev'])
                offspring_params['dropout_rate'] = round(round_with_step(perturbed_dropout_rate,
                                                                         self.config_params['dropout_rate']['min'],
                                                                         self.config_params['dropout_rate']['max'],
                                                                         self.config_params['dropout_rate']['step']), 4)
                parent_mutation['mutated_params']['dropout_rate'] = self.dropout_rate

        return offspring_id, CoDeepNEATModuleConv2DMaxPool2DDropout(config_params=self.config_params,
                                                                    module_id=offspring_id,
                                                                    parent_mutation=parent_mutation,
                                                                    **offspring_params)

    def create_crossover(self,
                         offspring_id,
                         less_fit_module,
                         max_degree_of_mutation) -> (int, CoDeepNEATModuleConv2DMaxPool2DDropout):
        """"""
        # Create offspring parameters by carrying over parameters of fitter parent for categorical parameters and
        # calculating parameter average between both modules for sortable parameters
        offspring_params = dict()

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': (self.module_id, less_fit_module.get_id()),
                           'mutation': 'crossover'}

        offspring_params['merge_method'] = self.merge_method
        offspring_params['filters'] = round_with_step(int((self.filters + less_fit_module.filters) / 2),
                                                      self.config_params['filters']['min'],
                                                      self.config_params['filters']['max'],
                                                      self.config_params['filters']['step'])
        offspring_params['kernel_size'] = self.kernel_size
        offspring_params['strides'] = self.strides
        offspring_params['padding'] = self.padding
        offspring_params['activation'] = self.activation
        offspring_params['kernel_init'] = self.kernel_init
        offspring_params['bias_init'] = self.bias_init
        offspring_params['max_pool_flag'] = self.max_pool_flag
        offspring_params['max_pool_size'] = self.max_pool_size
        offspring_params['dropout_flag'] = self.dropout_flag
        crossed_over_dropout_rate = round(round_with_step(((self.dropout_rate + less_fit_module.dropout_rate) / 2),
                                                          self.config_params['dropout_rate']['min'],
                                                          self.config_params['dropout_rate']['max'],
                                                          self.config_params['dropout_rate']['step']), 4)
        offspring_params['dropout_rate'] = crossed_over_dropout_rate

        return offspring_id, CoDeepNEATModuleConv2DMaxPool2DDropout(config_params=self.config_params,
                                                                    module_id=offspring_id,
                                                                    parent_mutation=parent_mutation,
                                                                    **offspring_params)

    def serialize(self) -> dict:
        """"""
        return {
            'module_type': 'Conv2DMaxPool2DDropout',
            'module_id': self.module_id,
            'parent_mutation': self.parent_mutation,
            'merge_method': self.merge_method,
            'filters': self.filters,
            'kernel_size': self.kernel_size,
            'strides': self.strides,
            'padding': self.padding,
            'activation': self.activation,
            'kernel_init': self.kernel_init,
            'bias_init': self.bias_init,
            'max_pool_flag': self.max_pool_flag,
            'max_pool_size': self.max_pool_size,
            'dropout_flag': self.dropout_flag,
            'dropout_rate': self.dropout_rate
        }
