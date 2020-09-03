import tensorflow as tf


class CoDeepNEATModel:
    def _create_model(self):
        """
        Create TF model from the supplied genotype. Create this model by iterating through the graph topology order that
        has been created by the blueprint. Create the associated module layers for each node and, if multiple
        connections are incoming, merge the inputs according to the associated module merging method. Append the
        genotype specified output layers to the output node of the ordered graph topology and assign the created model
        to self.model.
        """
        # Get preprocessed information from blueprint required for TF model creation
        node_species = self.blueprint.get_node_species()
        node_dependencies = self.blueprint.get_node_dependencies()
        graph_topology = self.blueprint.get_graph_topology()

        # Create the actual Tensorflow model through the functional keras API, starting with the inputs object and
        # saving the output of each layer in a dict that associates it with the node and serves for a later reference
        # in the functional style of model creation.
        inputs = tf.keras.Input(shape=self.input_shape, dtype=self.dtype)
        node_outputs = {1: inputs}

        # Create the TF model iteratively through the keras functional API by creating the single layers in the order in
        # which they are called, which is made possible through the topological sorting of the graph. Traverse this
        # topological sorting of the graph (though skip the first level as it always contains the Input node), process
        # the nodes of the level and then create the TF layers.
        for topology_level in graph_topology[1:]:
            for node in topology_level:
                # Determine the specific module of the current node and the required nodes the current node depends upon
                current_node_module = self.bp_assigned_modules[node_species[node]]
                current_node_dependencies = tuple(node_dependencies[node])

                # Determine if the node has multiple inputs and requires a merge. If so, configure it accordingly and
                # create merge input for the current node
                if len(current_node_dependencies) > 1:
                    # Get current node module merge method and deserialize it
                    merge_method = tf.keras.layers.deserialize(current_node_module.get_merge_method())

                    # Create list of all the nodes serving as input to the current node as well as their shapes. If a
                    # merge method with an axis has been supplied, disregard the values of that axis for potential
                    # downsampling
                    input_nodes = [node_outputs[node_dep] for node_dep in current_node_dependencies]
                    input_nodes_shapes = [list(in_node.shape) for in_node in input_nodes]
                    if hasattr(merge_method, 'axis'):
                        for shape in input_nodes_shapes:
                            shape[merge_method.axis] = None

                    # Assert that all out shapes of the input nodes are of the same dimension (only possible way
                    # CoDeepNEAT supports this) and determine if the output shapes of the input nodes are mismatched.
                    # If so, downsample. If not, merge.
                    output_dims = len(input_nodes_shapes[0])
                    assert all(len(shape) == output_dims for shape in input_nodes_shapes)
                    if not all(shape == input_nodes_shapes[0] for shape in input_nodes_shapes):
                        # Determine the smallest output shape to downsample to
                        smallest_out_shape = [None] * output_dims
                        for shape in input_nodes_shapes:
                            for i in range(output_dims):
                                if shape[i] is not None and (smallest_out_shape[i] is None
                                                             or shape[i] < smallest_out_shape[i]):
                                    smallest_out_shape[i] = shape[i]

                        # Create the list of input nodes with an additional downsampling layer for each mismatched input
                        input_nodes_downsampled = list()
                        for i in range(len(input_nodes)):
                            if input_nodes_shapes[i] != smallest_out_shape:
                                # output shape of input node mismatched. Create downsampling layer
                                ds_layer = current_node_module.create_downsampling_layer(in_shape=input_nodes[i].shape,
                                                                                         out_shape=smallest_out_shape)
                                downsampled_input = ds_layer(input_nodes[i])
                                input_nodes_downsampled.append(downsampled_input)
                            else:
                                # output shape of input node has minimal (downsampled) shape. Append the node output
                                # as is.
                                input_nodes_downsampled.append(input_nodes[i])

                        # Merge the downsampled input nodes, creating the input for the current node
                        node_input = merge_method(input_nodes_downsampled)

                    else:
                        # As the shapes of all input nodes are compatible, simply merge them to create the input for the
                        # current node
                        node_input = merge_method(input_nodes)

                else:
                    # As the current node only has 1 input, set this input node as the input for the current node
                    node_input = node_outputs[current_node_dependencies[0]]

                # Create the sequential layers of the module and pipe the just created input through this node/module
                node_layers = current_node_module.create_module_layers()
                node_out = node_input
                for layer in node_layers:
                    node_out = layer(node_out)

                # Register the final output of the sequential module layers as the output of the current node
                node_outputs[node] = node_out

        # Create the static output layers set by config and Pipe the results of the dynamic graph of modules through
        # them. The dynamic graph always has the output node 2, which is therefore the input to the output layers.
        deserialized_output_layers = [tf.keras.layers.deserialize(layer_config) for layer_config in self.output_layers]
        outputs = node_outputs[2]
        for out_layer in deserialized_output_layers:
            outputs = out_layer(outputs)

        # Create the complete keras Model through the functional API by identifying the inputs and output layers
        self.model = tf.keras.Model(inputs, outputs)
