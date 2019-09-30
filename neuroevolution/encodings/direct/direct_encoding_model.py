import numpy as np
import tensorflow as tf
from toposort import toposort

from .direct_encoding_gene import DirectEncodingConnection


class CustomWeightInputLayer(tf.keras.layers.Layer):
    def __init__(self, units, activation, kernel_weights, bias_weights, input_node_coords, trainable, dtype, dynamic):
        super(CustomWeightInputLayer, self).__init__(trainable=trainable, dtype=dtype, dynamic=dynamic)
        self.activation = activation
        self.kernel = self.add_weight(shape=kernel_weights.shape,
                                      dtype=self.dtype,
                                      initializer=tf.keras.initializers.deserialize('zeros'),
                                      trainable=self.trainable)
        self.bias = self.add_weight(shape=bias_weights.shape,
                                    dtype=self.dtype,
                                    initializer=tf.keras.initializers.deserialize('zeros'),
                                    trainable=self.trainable)
        self.set_weights([kernel_weights, bias_weights])
        self.built = True

        if len(input_node_coords) >= 2:
            self.input_node_coords = input_node_coords
            self.call = self._call_multiple_inputs
        else:
            (self.layer_index, self.node_index) = input_node_coords[0]
            self.call = self._call_single_inputs

    def _call_single_inputs(self, inputs, **kwargs):
        selected_inputs = inputs[self.layer_index][:, self.node_index:self.node_index + 1]
        return self.activation(tf.matmul(selected_inputs, self.kernel) + self.bias)

    def _call_multiple_inputs(self, inputs, **kwargs):
        selected_inputs = tf.concat(values=[inputs[layer_index][:, node_index:node_index + 1]
                                            for (layer_index, node_index) in self.input_node_coords], axis=1)
        return self.activation(tf.matmul(selected_inputs, self.kernel) + self.bias)


class DirectEncodingModel(tf.keras.Model):
    def __init__(self, genotype, trainable, dtype, run_eagerly):
        super(DirectEncodingModel, self).__init__(trainable=trainable, dtype=dtype)
        self.run_eagerly = run_eagerly

        nodes, connections, node_dependencies = self._create_gene_dicts(genotype)

        self.topology_dependency_levels = list(toposort(node_dependencies))

        node_to_topology = self._create_node_to_topology_mapping(self.topology_dependency_levels)

        self.custom_layers = []
        for _ in range(len(self.topology_dependency_levels) - 1):
            self.custom_layers.append([])

        for layer_index in range(len(self.custom_layers)):
            layer_node_dependencies = {node: node_dependencies[node]
                                       for node in self.topology_dependency_levels[layer_index + 1]}

            joined_layer_node_dependencies = self._join_keys_with_same_values(layer_node_dependencies)

            for joined_nodes, joined_nodes_input in joined_layer_node_dependencies.items():
                joined_nodes = list(joined_nodes)

                activation = nodes[joined_nodes[0]][1]
                assert all(nodes[node][1] == activation for node in joined_nodes)

                input_node_coords = [node_to_topology[node] for node in joined_nodes_input]

                kernel_weights = np.empty(shape=(len(input_node_coords), len(joined_nodes)), dtype=dtype.as_numpy_dtype)
                for column_index in range(kernel_weights.shape[1]):
                    for row_index in range(kernel_weights.shape[0]):
                        weight = connections[joined_nodes[column_index]][joined_nodes_input[row_index]]
                        kernel_weights[row_index, column_index] = weight

                bias_weights = np.empty(shape=(len(joined_nodes),), dtype=dtype.as_numpy_dtype)
                for node_index in range(len(joined_nodes)):
                    weight = nodes[joined_nodes[node_index]][0]
                    bias_weights[node_index] = weight

                nodes_function = CustomWeightInputLayer(units=len(joined_nodes),
                                                        activation=activation,
                                                        kernel_weights=kernel_weights,
                                                        bias_weights=bias_weights,
                                                        input_node_coords=input_node_coords,
                                                        trainable=trainable,
                                                        dtype=dtype,
                                                        dynamic=run_eagerly)
                self.custom_layers[layer_index].append(nodes_function)

    def call(self, inputs):
        inputs = [tf.cast(x=inputs, dtype=self.dtype)]
        for layers in self.custom_layers:
            layer_out = None
            for nodes_function in layers:
                out = nodes_function(inputs)
                layer_out = out if layer_out is None else tf.concat(values=[layer_out, out], axis=1)
            inputs.append(layer_out)
        return inputs[-1]

    @staticmethod
    def _create_gene_dicts(genotype):
        nodes = dict()
        connections = dict()
        node_dependencies = dict()
        for gene in genotype:
            if isinstance(gene, DirectEncodingConnection):
                conn_out = gene.conn_out
                conn_in = gene.conn_in
                if conn_out in connections:
                    connections[conn_out][conn_in] = gene.conn_weight
                    node_dependencies[conn_out].add(conn_in)
                else:
                    connections[conn_out] = {conn_in: gene.conn_weight}
                    node_dependencies[conn_out] = {conn_in}
            else:  # else gene is instance of DirectEncodingNode
                nodes[gene.node] = [gene.bias, gene.activation]
        return nodes, connections, node_dependencies

    @staticmethod
    def _create_node_to_topology_mapping(topology_dependency_levels):
        node_to_topology = dict()
        for layer_index in range(len(topology_dependency_levels)):
            layer_iterable = iter(topology_dependency_levels[layer_index])
            for node_index in range(len(topology_dependency_levels[layer_index])):
                node = next(layer_iterable)
                node_to_topology[node] = (layer_index, node_index)
        return node_to_topology

    @staticmethod
    def _join_keys_with_same_values(layer_node_dependencies):
        values_to_keys = dict()
        for k, v in layer_node_dependencies.items():
            frozen_v = frozenset(v)
            if frozen_v in values_to_keys:
                values_to_keys[frozen_v].add(k)
            else:
                values_to_keys[frozen_v] = {k}
        return {frozenset(v): list(k) for k, v in values_to_keys.items()}
