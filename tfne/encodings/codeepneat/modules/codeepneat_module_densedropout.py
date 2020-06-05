from __future__ import annotations

import math
import random

import numpy as np
import tensorflow as tf

from .codeepneat_module_base import CoDeepNEATModuleBase
from ....helper_functions import round_with_step


class CoDeepNEATModuleDenseDropout(CoDeepNEATModuleBase):
    """"""

    def __init__(self,
                 config_params,
                 module_id,
                 parent_mutation,
                 merge_method,
                 units,
                 activation,
                 kernel_init,
                 bias_init,
                 dropout_flag,
                 dropout_rate,
                 self_initialize=False):
        """"""
        # Register the dict listing the module parameter range specified in the config
        self.config_params = config_params

        # Register the module parameters
        super().__init__(module_id, parent_mutation)
        self.merge_method = merge_method
        self.units = units
        self.activation = activation
        self.kernel_init = kernel_init
        self.bias_init = bias_init
        self.dropout_flag = dropout_flag
        self.dropout_rate = dropout_rate

        # If self initialization flag is provided, initialize the module parameters as they are currently set to None
        if self_initialize:
            self.initialize()

    def __str__(self) -> str:
        """"""
        return "CoDeepNEAT DENSE Module | ID: {:>6} | Fitness: {:>6} | Units: {:>4} | Activ: {:>6} | Dropout: {:>4}" \
            .format('#' + str(self.module_id),
                    self.fitness,
                    self.units,
                    self.activation,
                    "None" if self.dropout_flag is False else self.dropout_rate)

    def create_module_layers(self, dtype) -> [tf.keras.layers.Layer, ...]:
        """"""
        # Create iterable that contains all layers concatenated in this module
        module_layers = list()

        # Create the basic keras Dense layer, needed in all variants of the module
        dense_layer = tf.keras.layers.Dense(units=self.units,
                                            activation=self.activation,
                                            kernel_initializer=self.kernel_init,
                                            bias_initializer=self.bias_init,
                                            dtype=dtype)
        module_layers.append(dense_layer)

        # If dropout flag present, add the dropout layer as configured to the module layers iterable
        if self.dropout_flag:
            dropout_layer = tf.keras.layers.Dropout(rate=self.dropout_rate,
                                                    dtype=dtype)
            module_layers.append(dropout_layer)

        # Return the iterable containing all layers present in the module
        return module_layers

    def create_downsampling_layer(self, in_shape, out_shape, dtype) -> tf.keras.layers.Layer:
        """"""
        raise NotImplementedError("Downsampling has not yet been implemented for DenseDropout Modules")

    def initialize(self):
        """"""
        # Uniformly randomly set module parameters
        self.merge_method = random.choice(self.config_params['merge_method'])
        random_units = random.randint(self.config_params['units']['min'],
                                      self.config_params['units']['max'])
        self.units = round_with_step(random_units,
                                     self.config_params['units']['min'],
                                     self.config_params['units']['max'],
                                     self.config_params['units']['step'])
        self.activation = random.choice(self.config_params['activation'])
        self.kernel_init = random.choice(self.config_params['kernel_init'])
        self.bias_init = random.choice(self.config_params['bias_init'])
        self.dropout_flag = random.random() < self.config_params['dropout_flag']
        random_dropout_rate = random.uniform(self.config_params['dropout_rate']['min'],
                                             self.config_params['dropout_rate']['max'])
        self.dropout_rate = round(round_with_step(random_dropout_rate,
                                                  self.config_params['dropout_rate']['min'],
                                                  self.config_params['dropout_rate']['max'],
                                                  self.config_params['dropout_rate']['step']), 4)

    def create_mutation(self,
                        offspring_id,
                        max_degree_of_mutation) -> (int, CoDeepNEATModuleDenseDropout):
        """"""
        # Copy the parameters of this parent module for the parameters of the offspring
        offspring_params = {'merge_method': self.merge_method,
                            'units': self.units,
                            'activation': self.activation,
                            'kernel_init': self.kernel_init,
                            'bias_init': self.bias_init,
                            'dropout_flag': self.dropout_flag,
                            'dropout_rate': self.dropout_rate}

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': self.module_id,
                           'mutation': 'mutation',
                           'mutated_params': dict()}

        # Determine exact integer amount of parameters to be mutated, though minimum is 1
        param_mutation_count = math.ceil(max_degree_of_mutation * 7)

        # Uniform randomly choose the parameters to be mutated
        parameters_to_mutate = random.sample(range(7), k=param_mutation_count)

        # Mutate offspring parameters. Categorical parameters are chosen randomly from all available values. Sortable
        # parameters are perturbed through a random normal distribution with the current value as mean and the config
        # specified stddev
        for param_to_mutate in parameters_to_mutate:
            if param_to_mutate == 0:
                offspring_params['merge_method'] = random.choice(self.config_params['merge_method'])
                parent_mutation['mutated_params']['merge_method'] = self.merge_method
            elif param_to_mutate == 1:
                perturbed_units = int(np.random.normal(loc=self.units,
                                                       scale=self.config_params['units']['stddev']))
                offspring_params['units'] = round_with_step(perturbed_units,
                                                            self.config_params['units']['min'],
                                                            self.config_params['units']['max'],
                                                            self.config_params['units']['step'])
                parent_mutation['mutated_params']['units'] = self.units
            elif param_to_mutate == 2:
                offspring_params['activation'] = random.choice(self.config_params['activation'])
                parent_mutation['mutated_params']['activation'] = self.activation
            elif param_to_mutate == 3:
                offspring_params['kernel_init'] = random.choice(self.config_params['kernel_init'])
                parent_mutation['mutated_params']['kernel_init'] = self.kernel_init
            elif param_to_mutate == 4:
                offspring_params['bias_init'] = random.choice(self.config_params['bias_init'])
                parent_mutation['mutated_params']['bias_init'] = self.bias_init
            elif param_to_mutate == 5:
                offspring_params['dropout_flag'] = not self.dropout_flag
                parent_mutation['mutated_params']['dropout_flag'] = self.dropout_flag
            else:  # param_to_mutate == 6:
                perturbed_dropout_rate = np.random.normal(loc=self.dropout_rate,
                                                          scale=self.config_params['dropout_rate']['stddev'])
                offspring_params['dropout_rate'] = round(round_with_step(perturbed_dropout_rate,
                                                                         self.config_params['dropout_rate']['min'],
                                                                         self.config_params['dropout_rate']['max'],
                                                                         self.config_params['dropout_rate']['step']), 4)
                parent_mutation['mutated_params']['dropout_rate'] = self.dropout_rate

        return offspring_id, CoDeepNEATModuleDenseDropout(config_params=self.config_params,
                                                          module_id=offspring_id,
                                                          parent_mutation=parent_mutation,
                                                          **offspring_params)

    def create_crossover(self,
                         offspring_id,
                         less_fit_module,
                         max_degree_of_mutation) -> (int, CoDeepNEATModuleDenseDropout):
        """"""
        # Create offspring parameters by carrying over parameters of fitter parent for categorical parameters and
        # calculating parameter average between both modules for sortable parameters
        offspring_params = dict()

        # Create the dict that keeps track of the mutations occuring for the offspring
        parent_mutation = {'parent_id': (self.module_id, less_fit_module.get_id()),
                           'mutation': 'crossover'}

        offspring_params['merge_method'] = self.merge_method
        offspring_params['units'] = round_with_step(int((self.units + less_fit_module.units) / 2),
                                                    self.config_params['units']['min'],
                                                    self.config_params['units']['max'],
                                                    self.config_params['units']['step'])
        offspring_params['activation'] = self.activation
        offspring_params['kernel_init'] = self.kernel_init
        offspring_params['bias_init'] = self.bias_init
        offspring_params['dropout_flag'] = self.dropout_flag
        offspring_params['dropout_rate'] = round(round_with_step((self.dropout_rate + less_fit_module.dropout_rate) / 2,
                                                                 self.config_params['dropout_rate']['min'],
                                                                 self.config_params['dropout_rate']['max'],
                                                                 self.config_params['dropout_rate']['step'], ), 4)

        return offspring_id, CoDeepNEATModuleDenseDropout(config_params=self.config_params,
                                                          module_id=offspring_id,
                                                          parent_mutation=parent_mutation,
                                                          **offspring_params)

    def serialize(self) -> dict:
        """"""
        return {
            'module_type': 'DenseDropout',
            'module_id': self.module_id,
            'parent_mutation': self.parent_mutation,
            'merge_method': self.merge_method,
            'units': self.units,
            'activation': self.activation,
            'kernel_init': self.kernel_init,
            'bias_init': self.bias_init,
            'dropout_flag': self.dropout_flag,
            'dropout_rate': self.dropout_rate
        }
