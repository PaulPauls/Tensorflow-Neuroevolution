import tensorflow as tf
from toposort import toposort


class CustomArbitraryInputNodeFunction(tf.keras.layers.Layer):
    """
    ToDo doc
    """

    def __init__(self, units, input_node_coords, activation, initializer_kernel, initializer_bias,
                 trainable, dtype, dynamic):
        super(CustomArbitraryInputNodeFunction, self).__init__(trainable=trainable, dtype=dtype, dynamic=dynamic)
        self.activation = activation

        # ToDo doc
        self.kernel = self.add_weight(shape=(len(input_node_coords), units),
                                      dtype=self.dtype,
                                      initializer=initializer_kernel,
                                      trainable=self.trainable)
        self.bias = self.add_weight(shape=(units,),
                                    dtype=self.dtype,
                                    initializer=initializer_bias,
                                    trainable=self.trainable)
        self.built = True

        # Set the layer's call function to either a function handling a single input or handling multiple inputs
        # and preconfigure the requirements to accelerate the heavily used call function.
        if len(input_node_coords) >= 2:
            self.input_node_coords = input_node_coords
            self.call = self.call_multiple_inputs
        else:
            (self.layer_index, self.node_index) = input_node_coords[0]
            self.call = self.call_single_inputs

    def call_single_inputs(self, inputs, **kwargs):
        selected_inputs = inputs[self.layer_index][:, self.node_index:self.node_index + 1]
        return self.activation(tf.matmul(selected_inputs, self.kernel) + self.bias)

    def call_multiple_inputs(self, inputs, **kwargs):
        selected_inputs = tf.concat(values=[inputs[layer_index][:, node_index:node_index + 1]
                                            for (layer_index, node_index) in self.input_node_coords], axis=1)
        return self.activation(tf.matmul(selected_inputs, self.kernel) + self.bias)


class DirectEncodingModel(tf.keras.Model):
    """
    ToDo doc
    """

    def __init__(self, genotype, activations, initializer_kernel, initializer_bias, trainable, dtype, run_eagerly):
        super(DirectEncodingModel, self).__init__(trainable=trainable, dtype=dtype)
        self.run_eagerly = run_eagerly

        # Create node_dependency dictionary. The keys are nodes that require preceding nodes to be evaluated beforehand.
        # The corresponding values of each key is the set of those nodes that need to be evalueated beforehand.

        node_dependencies = dict()
        for gene in genotype:
            key = gene.conn_out
            if key in node_dependencies.keys():
                node_dependencies[key].add(gene.conn_in)
            else:
                node_dependencies[key] = {gene.conn_in}

        # Create levels of dependency from the specified node_dependencies dict, showing which nodes can be evaluated
        # in parallel and which nodes need to be evaluated after all in the preceding levels have been.
        self.topology_dependency_levels = list(toposort(node_dependencies))

        # Create a translation dict that takes node as key and returns the topology coordinate tuple in the form
        # (layer_index, node_index_within_layer)
        node_to_topology = dict()
        for layer_index in range(len(self.topology_dependency_levels)):
            layer_iterable = iter(self.topology_dependency_levels[layer_index])
            for node_index in range(len(self.topology_dependency_levels[layer_index])):
                node = next(layer_iterable)
                node_to_topology[node] = (layer_index, node_index)

        # Create a list of lists of custom_layers. In the first dimension it traverses through layers, in the second
        # it traverses through the nodes in the respective layer.
        self.custom_layers = [None] * (len(self.topology_dependency_levels) - 1)

        # Traverse through each layer receiving an input from a preceding layer (therefore skip input layer and start
        # layer_index with 1) and create a 'joined_layer_node_dependencies dict in which the key is the set of nodes in
        # the layer that gets input from the nodes in the corresponding values.
        for layer_index in range(1, len(self.topology_dependency_levels)):
            layer_node_dependencies = {key: node_dependencies[key]
                                       for key in self.topology_dependency_levels[layer_index]}

            # Join all keys with the same values in a common joined key (which is a frozenset to be hashable)
            values_to_keys = dict()
            for k, v in layer_node_dependencies.items():
                frozen_v = frozenset(v)
                if frozen_v in values_to_keys:
                    values_to_keys[frozen_v].add(k)
                else:
                    values_to_keys[frozen_v] = {k}
            joined_layer_node_dependencies = {frozenset(v): set(k) for k, v in values_to_keys.items()}

            # Create CustomLayers for each joined node collection and add them to the double list of custom_layers
            self.custom_layers[layer_index - 1] = []
            for k, v in joined_layer_node_dependencies.items():
                input_node_coords = [node_to_topology[x] for x in v]
                activation = activations['out_activation'] if layer_index == len(self.topology_dependency_levels) - 1 \
                    else activations['default_activation']
                new_nodes_function = CustomArbitraryInputNodeFunction(units=len(k),
                                                                      input_node_coords=input_node_coords,
                                                                      activation=activation,
                                                                      initializer_kernel=initializer_kernel,
                                                                      initializer_bias=initializer_bias,
                                                                      trainable=self.trainable,
                                                                      dtype=self.dtype,
                                                                      dynamic=run_eagerly)
                self.custom_layers[layer_index - 1].append(new_nodes_function)

    def call(self, inputs, **kwargs):
        """
        ToDo doc
        :param inputs:
        :param kwargs:
        :return:
        """
        # tf.print("Model inputs: ", inputs)
        inputs = [tf.cast(x=inputs, dtype=self.dtype)]
        for layers in self.custom_layers:
            # tf.print("Layer inputs: ", inputs)
            layer_out = None
            for nodes_function in layers:
                out = nodes_function(inputs)
                # tf.print("nodes_function out: ", out)
                layer_out = out if layer_out is None else tf.concat(values=[layer_out, out], axis=1)
            # tf.print("layer_out: ", layer_out)
            inputs.append(layer_out)

        return inputs[-1]

    def get_topology_dependency_levels(self):
        return self.topology_dependency_levels
