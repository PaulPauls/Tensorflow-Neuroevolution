import tensorflow as tf
import tensorflow.keras.layers as tfkl
from toposort import toposort


class CustomLayer(tf.keras.layers.Layer):
    def __init__(self, num_outputs, input_node_coords, activation, dtype=tf.float64, **kwargs):
        super(CustomLayer, self).__init__(dtype=dtype, **kwargs)
        self.custom_layer = tfkl.Dense(units=num_outputs, activation=activation, dtype=dtype)
        self.input_node_coords = input_node_coords

    def call(self, input):
        # TODO
        if len(self.input_node_coords) >= 2:
            sel_input = tfkl.concatenate([input[layer][:, node:node+1] for (layer, node) in self.input_node_coords])
        else:
            (layer, node) = self.input_node_coords[0]
            sel_input = input[layer][:, node:node+1]
        # tf.print("sel_input: ", sel_input)
        out = self.custom_layer(sel_input)
        return out


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
            layer = list(self.topology_dependency_levels[layer_index])
            for node_index in range(len(layer)):
                node = layer[node_index]
                node_to_topology[node] = (layer_index, node_index)

        # Create the double list of custom_layers. In the first dimension it traverses through layers, in the second
        # it traverses through the nodes in the respective layer.
        self.custom_layers = [None] * (len(self.topology_dependency_levels) - 1)

        # Traverse through each layer receiving an input from a preceding layer (therefore skip input layer and start
        # layer_index with 1) and create a 'joined_layer_node_dependencies dict in which the key is the set of nodes in
        # the layer that gets input from the nodes in the corresponding values.
        for layer_index in range(1, len(self.topology_dependency_levels)):
            layer_node_dependencies = {key: node_dependencies[key] for key in
                                       node_dependencies.keys() & self.topology_dependency_levels[layer_index]}

            joined_layer_node_dependencies = self.join_keys_with_same_value(layer_node_dependencies)

            # Create CustomLayers for each joined node collection and add them to the double list of custom_layers
            self.custom_layers[layer_index-1] = []
            for k, v in joined_layer_node_dependencies.items():
                # TODO
                input_node_coords = [node_to_topology[x] for x in v]
                activation = activations['default_activation']
                new_layer = CustomLayer(num_outputs=len(k), input_node_coords=input_node_coords, activation=activation)
                self.custom_layers[layer_index-1].append(new_layer)

    def call(self, inputs):
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

    @staticmethod
    def join_keys_with_same_value(input_dict):
        # No general function but very specific to my translation function!
        # Joins all keys with the same values in a common joined key (as frozenset to be hashable)
        input_dict_frozen = {k: frozenset(v) for k, v in input_dict.items()}
        values_to_keys = dict()
        for k, v in input_dict_frozen.items():
            if v in values_to_keys.keys():
                values_to_keys[v].add(k)
            else:
                values_to_keys[v] = {k}

        joined_input_dict = {frozenset(v): set(k) for k, v in values_to_keys.items()}
        return joined_input_dict
