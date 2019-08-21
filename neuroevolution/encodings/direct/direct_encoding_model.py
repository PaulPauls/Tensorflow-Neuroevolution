import tensorflow as tf
import tensorflow.keras.layers as tfkl
from toposort import toposort


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, input_node_coords, activation, dtype=tf.float64):
        super(CustomLayer, self).__init__(dtype=dtype)
        self.custom_layer = tfkl.Dense(units=num_outputs, activation=activation, dtype=dtype)
        # Set the layer's call function to either a function handling a single input or handling multiple inputs
        # and preconfigure the requirements to accelerate the heavily used call function.
        if len(input_node_coords) >= 2:
            self.input_node_coords = input_node_coords
            self.call = self.call_multiple_inputs
        else:
            (self.layer_index, self.node_index) = input_node_coords[0]
            self.call = self.call_single_inputs

    def call_single_inputs(self, inputs, **kwargs):
        sel_input = inputs[self.layer_index][ :, self.node_index:self.node_index+1]
        return self.custom_layer(sel_input)

    def call_multiple_inputs(self, inputs, **kwargs):
        sel_input = tfkl.concatenate([inputs[layer_index][:, node_index:node_index+1] for (layer_index, node_index) in self.input_node_coords])
        return self.custom_layer(sel_input)

class DirectEncodingModel(tf.keras.Model):
    def __init__(self, genotype, activations, trainable):
        super(DirectEncodingModel, self).__init__(trainable=trainable)
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
            layer_node_dependencies = {key: node_dependencies[key] for key in self.topology_dependency_levels[layer_index]}

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
            self.custom_layers[layer_index-1] = []
            for k, v in joined_layer_node_dependencies.items():
                input_node_coords = [node_to_topology[x] for x in v]
                activation = activations['out_activation'] if layer_index == len(self.topology_dependency_levels)-1 else activations['default_activation']
                new_layer = CustomLayer(num_outputs=len(k), input_node_coords=input_node_coords, activation=activation)
                self.custom_layers[layer_index-1].append(new_layer)

    def call(self, inputs, **kwargs):
        # tf.print("Model Input: ", inputs)
        input_list = [tf.cast(inputs, tf.float64)]
        for layer_index in range(len(self.custom_layers)):
            # tf.print("Input List: ", input_list)
            layer = self.custom_layers[layer_index]
            layer_out = None
            for joined_nodes_index in range(len(layer)):
                out = layer[joined_nodes_index](input_list)
                if joined_nodes_index == 0:
                    layer_out = out
                else:
                    layer_out = tfkl.concatenate([layer_out, out])
            input_list.append(layer_out)
        return input_list[-1]

    def get_topology_dependency_levels(self):
        return self.topology_dependency_levels
