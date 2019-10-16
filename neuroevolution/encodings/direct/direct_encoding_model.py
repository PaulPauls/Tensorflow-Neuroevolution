import numpy as np
import tensorflow as tf
from toposort import toposort


def _process_genotype(genotype) -> (dict, dict, dict):
    """
    Process genotype by creating different dicts that contain all required information to create a Tensorflow model
    out of the genotype. Those three created dicts are explained in detail below.
    :param genotype: genotype dict with the keys being the gene-ids and the values being the genes
    :return: tuple of nodes dict, connections dict and node_dependencies dict
             nodes: dict with node (usually int) as dict key and a tuple of the node bias and activation function as
                    the dict value
             connections: dict representing each connection by associating each node (dict key) with the nodes it
                          receives input from and the weight of their connection (dict value). The conn_out of the
                          connection gene is the dict key and a seperate dict, associating the conn_in with the
                          conn_weight of the connection gene, is the dict value.
             node_dependencies: dict associating each node (dict key) with a set of the nodes (dict value) it
                                receives input from
    """
    nodes = dict()
    connections = dict()
    node_dependencies = dict()
    for gene in genotype.values():
        try:  # If gene isinstance of DirectEncodingConnection:
            # skip gene if it is disabled
            if not gene.enabled:
                continue
            conn_out = gene.conn_out
            conn_in = gene.conn_in
            if conn_out in connections:
                connections[conn_out][conn_in] = gene.conn_weight
                node_dependencies[conn_out].add(conn_in)
            else:
                connections[conn_out] = {conn_in: gene.conn_weight}
                node_dependencies[conn_out] = {conn_in}
        except AttributeError:  # else (gene isinstance of DirectEncodingNode):
            nodes[gene.node] = (gene.bias, gene.activation)
    return nodes, connections, node_dependencies


def _create_node_coordinates(topology_levels) -> dict:
    """
    Create and return a dict associating each node (dict key) with their coordinate value (dict value) in the form
    of (layer_index, node_index) as they are organized in the supplied topology_level parameter.
    :param topology_levels: tuple with each element specifying the set of nodes that have to be precomputed before
                            the next element's set of nodes can be computed, as they serve as input nodes to this
                            next element's set of nodes
    :return: dict associating the nodes in topology_levels (dict key) with the coordinates in the topology_levels
             (dict value)
    """
    node_coordinates = dict()
    for layer_index in range(len(topology_levels)):
        layer_iterable = iter(topology_levels[layer_index])
        for node_index in range(len(topology_levels[layer_index])):
            node = next(layer_iterable)
            node_coordinates[node] = (layer_index, node_index)
    return node_coordinates


def _join_keys(node_dependencies) -> dict:
    """
    Recreate node_dependencies parameter dict by using frozensets as keys consisting of all keys that have the same
    dict value. Also convert the dict value of the input parameter to tuple. Return this converted dict.
    :param node_dependencies: dict associating each node (dict key) with a set of the nodes (dict value) it receives
                              input from
    :return: dict associating the frozenset of all keys with the same dict value with this dict value.
    """
    values_to_keys = dict()
    for k, v in node_dependencies.items():
        frozen_v = frozenset(v)
        if frozen_v in values_to_keys:
            values_to_keys[frozen_v].add(k)
        else:
            values_to_keys[frozen_v] = {k}
    return {frozenset(v): tuple(k) for k, v in values_to_keys.items()}


class CustomWeightAndInputLayerTrainable(tf.keras.layers.Layer):
    """
    Custom Tensorflow layer that allows for arbitrary input nodes from any layer as well as custom kernel and bias
    weight setting. The arbitrariness of the input nodes is made possible through usage of coordinates specifying the
    layer and exact node in that layer for every input node for the CustomWeightAndInputLayer. The layer is fully
    compatible with the rest of the Tensorflow infrastructure and supports static-graph building, auto-gradient, etc.
    """
    initializer = tf.keras.initializers.zeros()

    def __init__(self, activation, kernel_weights, bias_weights, input_node_coords, dtype, dynamic):
        super(CustomWeightAndInputLayerTrainable, self).__init__(trainable=True, dtype=dtype, dynamic=dynamic)
        self.activation = activation
        self.kernel = self.add_weight(shape=kernel_weights.shape,
                                      dtype=self.dtype,
                                      initializer=CustomWeightAndInputLayerTrainable.initializer,
                                      trainable=True)
        self.bias = self.add_weight(shape=bias_weights.shape,
                                    dtype=self.dtype,
                                    initializer=CustomWeightAndInputLayerTrainable.initializer,
                                    trainable=True)
        self.set_weights((kernel_weights, bias_weights))
        self.built = True

        if len(input_node_coords) >= 2:
            self.input_node_coords = input_node_coords
            self.call = self._call_multiple_inputs
        else:
            (self.layer_index, self.node_index) = input_node_coords[0]
            self.call = self._call_single_inputs

    def _call_single_inputs(self, inputs, **kwargs) -> tf.Tensor:
        """
        Layer call, whereby the layer has only a single input node
        :param inputs: array of Tensorflow tensors representing the output of each preceding layer
        :return: Tensorflow tensor of the computed layer results
        """
        selected_inputs = inputs[self.layer_index][:, self.node_index:self.node_index + 1]
        return self.activation(tf.matmul(selected_inputs, self.kernel) + self.bias)

    def _call_multiple_inputs(self, inputs, **kwargs) -> tf.Tensor:
        """
        Layer call, whereby the layer has more than one input nodes
        :param inputs: array of Tensorflow tensors representing the output of each preceding layer
        :return: Tensorflow tensor of the computed layer results
        """
        selected_inputs = tf.concat(values=[inputs[layer_index][:, node_index:node_index + 1]
                                            for (layer_index, node_index) in self.input_node_coords], axis=1)
        return self.activation(tf.matmul(selected_inputs, self.kernel) + self.bias)


class DirectEncodingModelTrainable(tf.keras.Model):
    """
    Tensorflow model that builds a (exclusively) feed-forward topology with custom set connection weights and node
    biases/activations from the supplied genotype in the constructor. The built Tensorflow model is fully compatible
    with the rest of the Tensorflow infrastructure and supports static-graph building, auto-gradient, etc
    """

    def __init__(self, genotype, dtype, run_eagerly):
        """
        Creates the trainable feed-forward Tensorflow model out of the supplied genotype with custom parameters
        :param genotype: genotype dict with the keys being the gene-ids and the values being the genes
        :param dtype: Tensorflow datatype of the model
        :param run_eagerly: bool flag if model should be run eagerly (by CPU) or if static GPU graph should be build
        """
        super(DirectEncodingModelTrainable, self).__init__(trainable=True, dtype=dtype)
        self.run_eagerly = run_eagerly

        nodes, connections, node_dependencies = _process_genotype(genotype)

        self.topology_levels = tuple(toposort(node_dependencies))
        node_coordinates = _create_node_coordinates(self.topology_levels)

        self.custom_layers = [[] for _ in range(len(self.topology_levels) - 1)]
        for layer_index in range(len(self.custom_layers)):
            # Create node_dependencies specific for the current layer and with joined keys (conn_outs) if the have the
            # same input values (conn_ins)
            layer_node_dependencies = {node: node_dependencies[node] for node in self.topology_levels[layer_index + 1]}
            joined_layer_node_dependencies = _join_keys(layer_node_dependencies)

            for joined_nodes, joined_nodes_input in joined_layer_node_dependencies.items():
                joined_nodes = tuple(joined_nodes)

                activation = nodes[joined_nodes[0]][1]
                # Assert that all nodes for which the same CustomWeightAndInputLayer is created have the same activation
                assert all(nodes[node][1] == activation for node in joined_nodes)

                input_node_coords = [node_coordinates[node] for node in joined_nodes_input]

                # Create custom kernel weight matrix from connection weights supplied in genotype
                kernel_weights = np.empty(shape=(len(input_node_coords), len(joined_nodes)), dtype=dtype.as_numpy_dtype)
                for column_index in range(kernel_weights.shape[1]):
                    for row_index in range(kernel_weights.shape[0]):
                        weight = connections[joined_nodes[column_index]][joined_nodes_input[row_index]]
                        kernel_weights[row_index, column_index] = weight

                # Create custom bias weight matrix from bias weights supplied in genotype
                bias_weights = np.empty(shape=(len(joined_nodes),), dtype=dtype.as_numpy_dtype)
                for node_index in range(len(joined_nodes)):
                    weight = nodes[joined_nodes[node_index]][0]
                    bias_weights[node_index] = weight

                # Create nodes function for those joined nodes that have the same input as their value can be computed
                # in unison
                nodes_function = CustomWeightAndInputLayerTrainable(activation=activation,
                                                                    kernel_weights=kernel_weights,
                                                                    bias_weights=bias_weights,
                                                                    input_node_coords=input_node_coords,
                                                                    dtype=dtype,
                                                                    dynamic=run_eagerly)
                self.custom_layers[layer_index].append(nodes_function)

    def call(self, inputs) -> np.ndarray:
        """
        Model call of the DirectEncoding feed-forward model with arbitrarily connected nodes. The output of each layer
        is continually preserved, concatenated and then supplied together with the ouputs of all preceding layers to the
        next layer.
        :param inputs: Tensorflow or numpy array of one or multiple inputs to predict the output for
        :return: numpy array representing the predicted output to the input
        """
        inputs = [tf.cast(x=inputs, dtype=self.dtype)]
        for layers in self.custom_layers:
            layer_out = None
            for nodes_function in layers:
                out = nodes_function(inputs)
                layer_out = out if layer_out is None else tf.concat(values=[layer_out, out], axis=1)
            inputs.append(layer_out)
        return inputs[-1]


class CustomWeightAndInputLayerNontrainable:
    """
    Custom sparsely connected layer that allows for arbitrary input nodes from any layer, multiplying the inputs with
    the custom set kernel and bias. The arbitrariness of the input nodes is made possible through usage of coordinates
    specifying the layer and exact node in that layer for every input node. The layer is not trainable and even though
    it uses Tensorflow functionality is not compatible with the rest of the Tensorflow infrastructure.
    """

    def __init__(self, activation, kernel, bias, input_node_coords, dtype):
        self.activation = activation
        self.kernel = kernel
        self.bias = bias
        self.input_node_coords = input_node_coords
        self.dtype = dtype

    def __call__(self, inputs) -> tf.Tensor:
        """
        Layer call, whereby the size of the input nodes is determined and then accordingly multiplied with the kernel,
        added with the bias and the activation function is applied.
        :param inputs: array of Tensorflow or numpy tensors representing the output of each preceding layer
        :return: Tensorflow tensor of the computed layer results
        """
        if len(self.input_node_coords) >= 2:
            selected_inputs = np.concatenate([inputs[layer_index][:, node_index:node_index + 1]
                                              for (layer_index, node_index) in self.input_node_coords], axis=1)
            return self.activation(np.matmul(selected_inputs, self.kernel) + self.bias)
        else:
            layer_index, node_index = self.input_node_coords[0]
            selected_inputs = inputs[layer_index][:, node_index:node_index + 1]
            return self.activation(np.matmul(selected_inputs, self.kernel) + self.bias)


class DirectEncodingModelNontrainable:
    """
    Neural Network model that builds a (exclusively) feed-forward topology with custom set connection weights and node
    biases/activations from the supplied genotype in the constructor. The built model is non trainable and not
    compatible with the rest of the Tensorflow infrastructure.
    """

    def __init__(self, genotype, dtype):
        """
        Creates the non-trainable feed-forward model out of the supplied genotype with custom parameters
        :param genotype: genotype dict with the keys being the gene-ids and the values being the genes
        :param dtype: Tensorflow datatype of the model
        """
        self.dtype = dtype.as_numpy_dtype

        nodes, connections, node_dependencies = _process_genotype(genotype)

        self.topology_levels = tuple(toposort(node_dependencies))
        node_coordinates = _create_node_coordinates(self.topology_levels)

        self.custom_layers = [[] for _ in range(len(self.topology_levels) - 1)]
        for layer_index in range(len(self.custom_layers)):
            # Create node_dependencies specific for the current layer and with joined keys (conn_outs) if the have the
            # same input values (conn_ins)
            layer_node_dependencies = {node: node_dependencies[node] for node in self.topology_levels[layer_index + 1]}
            joined_layer_node_dependencies = _join_keys(layer_node_dependencies)

            for joined_nodes, joined_nodes_input in joined_layer_node_dependencies.items():
                joined_nodes = tuple(joined_nodes)

                activation = nodes[joined_nodes[0]][1]
                # Assert that all nodes for which the same CustomWeightAndInputLayer is created have the same activation
                assert all(nodes[node][1] == activation for node in joined_nodes)

                input_node_coords = [node_coordinates[node] for node in joined_nodes_input]

                # Create custom kernel weight matrix from connection weights supplied in genotype
                kernel = np.empty(shape=(len(input_node_coords), len(joined_nodes)), dtype=self.dtype)
                for column_index in range(kernel.shape[1]):
                    for row_index in range(kernel.shape[0]):
                        weight = connections[joined_nodes[column_index]][joined_nodes_input[row_index]]
                        kernel[row_index, column_index] = weight

                # Create custom bias weight matrix from bias weights supplied in genotype
                bias = np.empty(shape=(len(joined_nodes),), dtype=self.dtype)
                for node_index in range(len(joined_nodes)):
                    weight = nodes[joined_nodes[node_index]][0]
                    bias[node_index] = weight

                # Create nodes function for those joined nodes that have the same input as their value can be computed
                # in unison
                nodes_function = CustomWeightAndInputLayerNontrainable(activation=activation,
                                                                       kernel=kernel,
                                                                       bias=bias,
                                                                       input_node_coords=input_node_coords,
                                                                       dtype=dtype)
                self.custom_layers[layer_index].append(nodes_function)

    def predict(self, inputs) -> tf.Tensor:
        """
        Model call of the DirectEncoding feed-forward model with arbitrarily connected nodes. The output of each layer
        is continually preserved, concatenated and then supplied together with the ouputs of all preceding layers to the
        next layer.
        :param inputs: Tensorflow or numpy array of one or multiple inputs to predict the output for
        :return: Tensorflow tensor representing the predicted output to the input
        """
        inputs = [inputs.astype(self.dtype)]
        for layers in self.custom_layers:
            layer_out = None
            for nodes_function in layers:
                out = nodes_function(inputs)
                layer_out = out if layer_out is None else np.concatenate((layer_out, out), axis=1)
            inputs.append(layer_out)
        return inputs[-1]
