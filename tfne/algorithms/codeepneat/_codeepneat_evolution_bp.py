import math
import random

from tfne.helper_functions import round_with_step
from tfne.encodings.codeepneat.codeepneat_blueprint import CoDeepNEATBlueprintNode, CoDeepNEATBlueprintConn


class CoDeepNEATEvolutionBP:
    def _evolve_blueprints(self, bp_spec_offspring, bp_spec_parents, mod_spec_extinct) -> [int]:
        """"""
        ### Evolve Species of Elite Blueprints ###
        # Check if node module species of the elite blueprints belong to extinct module species. If so, create a
        # node module species mutation of the elite blueprints and replace the elite blueprints.
        for spec_id, spec_bp_ids in self.pop.bp_species.items():
            orig_spec_bp_ids = spec_bp_ids.copy()
            for bp_id in orig_spec_bp_ids:
                # Compare blueprint node module species with extinct module species. If no intersection, continue on
                bp_mod_species = self.pop.blueprints[bp_id].get_species()
                bp_mod_extinct_species = bp_mod_species.intersection(mod_spec_extinct)
                if not bp_mod_extinct_species:
                    continue

                # Create new mutated blueprint with elite blueprint as parent and only the extinct node module species
                # changed.
                new_bp_id, new_bp = self._create_mutated_blueprint_node_spec(self.pop.blueprints[bp_id],
                                                                             0,
                                                                             mod_spec_extinct)
                # Add newly created blueprint to blueprint container, replace old bp id with new bp id in the species
                # association and then remove the old blueprint
                self.pop.blueprints[new_bp_id] = new_bp
                self.pop.bp_species[spec_id].remove(bp_id)
                bp_spec_parents[spec_id].remove(bp_id)
                self.pop.bp_species[spec_id].append(new_bp_id)
                bp_spec_parents[spec_id].append(new_bp_id)
                if self.bp_spec_type != 'basic' and self.pop.bp_species_repr[spec_id] == bp_id:
                    self.pop.bp_species_repr[spec_id] = new_bp_id
                del self.pop.blueprints[bp_id]

        #### Evolve Blueprints ####
        # Create container for new blueprints that will be speciated in a later function
        new_blueprint_ids = list()

        # Calculate the brackets for a random float to fall into in order to choose a specific evolutionary method
        bp_mutation_add_node_bracket = self.bp_mutation_add_conn_prob + self.bp_mutation_add_node_prob
        bp_mutation_rem_conn_bracket = bp_mutation_add_node_bracket + self.bp_mutation_rem_conn_prob
        bp_mutation_rem_node_bracket = bp_mutation_rem_conn_bracket + self.bp_mutation_rem_node_prob
        bp_mutation_node_spec_bracket = bp_mutation_rem_node_bracket + self.bp_mutation_node_spec_prob
        bp_mutation_optimizer_bracket = bp_mutation_node_spec_bracket + self.bp_mutation_optimizer_prob

        # Traverse through each species and create according amount of offspring as determined prior during selection
        for spec_id, species_offspring in bp_spec_offspring.items():
            if spec_id == 'reinit':
                continue

            for _ in range(species_offspring):
                # Choose random float vaue determining specific evolutionary method to evolve the chosen blueprint.
                # Choose random parent from list of elligible parents for the species
                random_choice = random.random()
                parent_blueprint = self.pop.blueprints[random.choice(bp_spec_parents[spec_id])]

                # If randomly chosen parent for mutation contains extinct species, force node module species mutation
                if parent_blueprint.get_species().intersection(mod_spec_extinct):
                    random_choice = (bp_mutation_rem_node_bracket + bp_mutation_node_spec_bracket) / 2.0

                if random_choice < self.bp_mutation_add_conn_prob:
                    ## Create new blueprint by adding connection ##
                    max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                    new_bp_id, new_bp = self._create_mutated_blueprint_add_conn(parent_blueprint,
                                                                                max_degree_of_mutation)

                elif random_choice < bp_mutation_add_node_bracket:
                    ## Create new blueprint by adding node ##
                    max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                    new_bp_id, new_bp = self._create_mutated_blueprint_add_node(parent_blueprint,
                                                                                max_degree_of_mutation)

                elif random_choice < bp_mutation_rem_conn_bracket:
                    ## Create new blueprint by removing connection ##
                    max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                    new_bp_id, new_bp = self._create_mutated_blueprint_rem_conn(parent_blueprint,
                                                                                max_degree_of_mutation)

                elif random_choice < bp_mutation_rem_node_bracket:
                    ## Create new blueprint by removing node ##
                    max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                    new_bp_id, new_bp = self._create_mutated_blueprint_rem_node(parent_blueprint,
                                                                                max_degree_of_mutation)

                elif random_choice < bp_mutation_node_spec_bracket:
                    ## Create new blueprint by mutating species in nodes ##
                    max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                    new_bp_id, new_bp = self._create_mutated_blueprint_node_spec(parent_blueprint,
                                                                                 max_degree_of_mutation,
                                                                                 mod_spec_extinct)

                elif random_choice < bp_mutation_optimizer_bracket:
                    ## Create new blueprint by mutating the associated optimizer ##
                    new_bp_id, new_bp = self._create_mutated_blueprint_optimizer(parent_blueprint)

                else:  # random_choice < bp_crossover_bracket:
                    ## Create new blueprint through crossover ##
                    # Try randomly selecting another parent blueprint and checking that other parnet bp for extinct
                    # node module species. If species has only 1 member or other blueprint has extinct node module
                    # species fail and create a new blueprint by adding a node to the original parent blueprint.
                    try:
                        other_bp_id_pool = bp_spec_parents[spec_id].copy()
                        other_bp_id_pool.remove(parent_blueprint.get_id())
                        other_bp_id = random.choice(other_bp_id_pool)
                        other_bp = self.pop.blueprints[other_bp_id]
                        if other_bp.get_species().intersection(mod_spec_extinct):
                            raise IndexError
                    except IndexError:
                        max_degree_of_mutation = random.uniform(1e-323, self.bp_max_mutation)
                        new_bp_id, new_bp = self._create_mutated_blueprint_add_node(parent_blueprint,
                                                                                    max_degree_of_mutation)
                    else:
                        # Create crossover blueprint if second valid blueprint was found
                        new_bp_id, new_bp = self._create_crossed_over_blueprint(parent_blueprint,
                                                                                other_bp)

                # Add newly created blueprint to the bp container and to the list of bps that have to be speciated
                self.pop.blueprints[new_bp_id] = new_bp
                new_blueprint_ids.append(new_bp_id)

        #### Reinitialize Blueprints ####
        # Initialize predetermined number of new blueprints as species went extinct and reinitialization is activated
        if 'reinit' in bp_spec_offspring:
            available_mod_species = tuple(self.pop.mod_species.keys())
            for _ in range(bp_spec_offspring['reinit']):
                # Determine the module species of the initial (and only) node
                initial_node_species = random.choice(available_mod_species)

                # Initialize a new blueprint with minimal graph only using initial node species
                new_bp_id, new_bp = self._create_initial_blueprint(initial_node_species)

                # Add newly created blueprint to the bp container and to the list of bps that have to be speciated
                self.pop.blueprints[new_bp_id] = new_bp
                new_blueprint_ids.append(new_bp_id)

        # Return the list of new blueprint ids for later speciation as well as the updated list of blueprint parents
        # for each species
        return new_blueprint_ids, bp_spec_parents

    def _create_mutated_blueprint_add_conn(self, parent_blueprint, max_degree_of_mutation):
        """"""
        # Copy the parameters of the parent blueprint and get the pre-analyzed topology of the parent graph
        blueprint_graph, optimizer_factory = parent_blueprint.copy_parameters()
        bp_graph_topology = parent_blueprint.get_graph_topology()

        # Create the dict that keeps track of the way a blueprint has been mutated
        parent_mutation = {'parent_id': parent_blueprint.get_id(),
                           'mutation': 'add_conn',
                           'added_genes': list()}

        # Create collections of all nodes and present connections in the copied blueprint graph
        bp_graph_conns = dict()
        bp_graph_nodes = list()
        for gene in blueprint_graph.values():
            if isinstance(gene, CoDeepNEATBlueprintNode):
                bp_graph_nodes.append(gene.node)
            else:  # isinstance(gene, CoDeepNEATBlueprintConn)
                bp_graph_conns[(gene.conn_start, gene.conn_end, gene.enabled)] = gene.gene_id

        # Remove end-node (node 2) from this list and shuffle it, as it later serves to randomly pop the start node of
        # the newly created connection
        bp_graph_nodes.remove(2)
        random.shuffle(bp_graph_nodes)

        # Determine specifically how many connections will be added
        number_of_conns_to_add = math.ceil(max_degree_of_mutation * len(bp_graph_conns))

        # Add connections in a loop until predetermined number of connections that are to be added is reached or until
        # the possible starting nodes run out
        added_conns_counter = 0
        while added_conns_counter < number_of_conns_to_add and len(bp_graph_nodes) > 0:
            # Choose random start node from all possible nodes by popping it from a preshuffled list of graph nodes
            start_node = bp_graph_nodes.pop()

            # Determine the list of all possible end nodes that are behind the start node as implementation currently
            # only supports feedforward topologies. Then shuffle the end nodes as they will later be randomly popped
            start_node_level = None
            for i in range(len(bp_graph_topology)):
                if start_node in bp_graph_topology[i]:
                    start_node_level = i
                    break
            possible_end_nodes = list(set().union(*bp_graph_topology[start_node_level + 1:]))
            random.shuffle(possible_end_nodes)

            # Traverse all possible end nodes randomly and create and add a blueprint connection to the offspring
            # blueprint graph if the specific connection tuple is not yet present.
            while len(possible_end_nodes) > 0:
                end_node = possible_end_nodes.pop()

                # If connection already present and enabled in graph, move on to the next possible end node
                if (start_node, end_node, True) in bp_graph_conns:
                    continue

                # If connection already present but disabled in graph, activate it
                elif (start_node, end_node, False) in bp_graph_conns:
                    gene_id = bp_graph_conns[(start_node, end_node, False)]
                    blueprint_graph[gene_id].set_enabled(True)
                    parent_mutation['added_genes'].append(gene_id)
                    added_conns_counter += 1

                # If connection not yet present in graph, create it
                else:  # (start_node, end_node) not in bp_graph_conns:
                    gene_id, gene = self.enc.create_blueprint_conn(conn_start=start_node,
                                                                   conn_end=end_node)
                    blueprint_graph[gene_id] = gene
                    parent_mutation['added_genes'].append(gene_id)
                    added_conns_counter += 1

        # Create and return the offspring blueprint with the edited blueprint graph having additional connections as
        # well as the parent duplicated optimizer factory.
        return self.enc.create_blueprint(blueprint_graph=blueprint_graph,
                                         optimizer_factory=optimizer_factory,
                                         parent_mutation=parent_mutation)

    def _create_mutated_blueprint_add_node(self, parent_blueprint, max_degree_of_mutation):
        """"""
        # Copy the parameters of the parent blueprint for the offspring
        blueprint_graph, optimizer_factory = parent_blueprint.copy_parameters()

        # Create the dict that keeps track of the way a blueprint has been mutated
        parent_mutation = {'parent_id': parent_blueprint.get_id(),
                           'mutation': 'add_node',
                           'added_genes': list()}

        # Analyze amount of nodes already present in bp graph as well as collect all gene ids of the present connections
        # that can possibly be split
        node_count = 0
        bp_graph_conn_ids = list()
        for gene in blueprint_graph.values():
            if isinstance(gene, CoDeepNEATBlueprintNode):
                node_count += 1
            elif gene.enabled:  # and isinstance(gene, CoDeepNEATBlueprintConn)
                # Only consider a connection for bp_graph_conn_ids if it is enabled
                bp_graph_conn_ids.append(gene.gene_id)

        # Determine how many nodes will be added, which connection gene_ids will be split for that and what possible
        # species can be assigned to those new nodes
        number_of_nodes_to_add = math.ceil(max_degree_of_mutation * node_count)
        gene_ids_to_split = random.sample(bp_graph_conn_ids, k=number_of_nodes_to_add)
        available_mod_species = tuple(self.pop.mod_species.keys())

        # Split all chosen connections by setting them to disabled, querying the new node id from the encoding and then
        # creating the new node and the associated 2 connections through the encoding.
        for gene_id_to_split in gene_ids_to_split:
            # Determine start and end node of connection and disable it
            conn_start = blueprint_graph[gene_id_to_split].conn_start
            conn_end = blueprint_graph[gene_id_to_split].conn_end
            blueprint_graph[gene_id_to_split].set_enabled(False)

            # Create a new unique node if connection has not yet been split by any other mutation. Otherwise create the
            # same node. Choose species for new node randomly.
            new_node = self.enc.create_node_for_split(conn_start, conn_end)
            new_species = random.choice(available_mod_species)

            # Create the node and connections genes for the new node addition and add them to the blueprint graph
            gene_id, gene = self.enc.create_blueprint_node(node=new_node, species=new_species)
            blueprint_graph[gene_id] = gene
            parent_mutation['added_genes'].append(gene_id)
            gene_id, gene = self.enc.create_blueprint_conn(conn_start=conn_start, conn_end=new_node)
            blueprint_graph[gene_id] = gene
            parent_mutation['added_genes'].append(gene_id)
            gene_id, gene = self.enc.create_blueprint_conn(conn_start=new_node, conn_end=conn_end)
            blueprint_graph[gene_id] = gene
            parent_mutation['added_genes'].append(gene_id)

        # Create and return the offspring blueprint with the edited blueprint graph having a new node through a split
        # connection as well as the parent duplicated optimizer factory.
        return self.enc.create_blueprint(blueprint_graph=blueprint_graph,
                                         optimizer_factory=optimizer_factory,
                                         parent_mutation=parent_mutation)

    def _create_mutated_blueprint_rem_conn(self, parent_blueprint, max_degree_of_mutation):
        """"""
        # Copy the parameters of the parent blueprint for the offspring
        blueprint_graph, optimizer_factory = parent_blueprint.copy_parameters()

        # Create the dict that keeps track of the way a blueprint has been mutated
        parent_mutation = {'parent_id': parent_blueprint.get_id(),
                           'mutation': 'rem_conn',
                           'removed_genes': list()}

        # Analyze blueprint graph for the amount of connections present as well as the amount of incoming and outgoing
        # connections for each node. Add connections that are disabled automatically to the list of possibly removable
        # genes
        conn_count = 0
        enabled_conn_ids = list()
        removable_gene_ids = list()
        node_incoming_conns_count = dict()
        node_outgoing_conns_count = dict()
        for gene in blueprint_graph.values():
            if isinstance(gene, CoDeepNEATBlueprintConn):
                conn_count += 1
                if not gene.enabled:
                    removable_gene_ids.append(gene.gene_id)
                    continue
                enabled_conn_ids.append(gene.gene_id)
                # Create counter of incoming connections for each node
                if gene.conn_end in node_incoming_conns_count:
                    node_incoming_conns_count[gene.conn_end] += 1
                else:
                    node_incoming_conns_count[gene.conn_end] = 1
                # Create counter of outgoing connections for each node
                if gene.conn_start in node_outgoing_conns_count:
                    node_outgoing_conns_count[gene.conn_start] += 1
                else:
                    node_outgoing_conns_count[gene.conn_start] = 1

        # Randomly check the enabled connections if they are the only incoming or outgoing connection of a node. If so,
        # don't consider them as potentially removable as the removable of the connecction would also effectively remove
        # a node. If the node has additional connections, consider the connection as removable.
        random.shuffle(enabled_conn_ids)
        for conn_id in enabled_conn_ids:
            gene = blueprint_graph[conn_id]
            if node_incoming_conns_count[gene.conn_end] > 1 and node_outgoing_conns_count[gene.conn_start] > 1:
                removable_gene_ids.append(gene.gene_id)
                node_incoming_conns_count[gene.conn_end] -= 1
                node_outgoing_conns_count[gene.conn_start] -= 1

        # Determine how many conns will be removed based on the total connection count, including disabled connections
        number_of_conns_to_rem = math.ceil(max_degree_of_mutation * conn_count)
        if number_of_conns_to_rem > len(removable_gene_ids):
            number_of_conns_to_rem = len(removable_gene_ids)
        gene_ids_to_remove = random.sample(removable_gene_ids, k=number_of_conns_to_rem)

        # Remove determined genes from the offspring blueprint graph and note the mutation
        for gene_id_to_remove in gene_ids_to_remove:
            del blueprint_graph[gene_id_to_remove]
            parent_mutation['removed_genes'].append(gene_id_to_remove)

        # Create and return the offspring blueprint with the edited blueprint graph having one or multiple connections
        # removed though still having at least 1 connection to and from each node.
        return self.enc.create_blueprint(blueprint_graph=blueprint_graph,
                                         optimizer_factory=optimizer_factory,
                                         parent_mutation=parent_mutation)

    def _create_mutated_blueprint_rem_node(self, parent_blueprint, max_degree_of_mutation):
        """"""
        # Copy the parameters of the parent blueprint for the offspring
        blueprint_graph, optimizer_factory = parent_blueprint.copy_parameters()

        # Create the dict that keeps track of the way a blueprint has been mutated
        parent_mutation = {'parent_id': parent_blueprint.get_id(),
                           'mutation': 'rem_node',
                           'removed_genes': list()}

        # Collect all gene_ids of nodes that are not the input or output node (as they are unremovable) and
        # shuffle the list of those node ids for later random popping.
        node_count = 0
        bp_graph_node_ids = list()
        for gene in blueprint_graph.values():
            if isinstance(gene, CoDeepNEATBlueprintNode):
                node_count += 1
                if gene.node != 1 and gene.node != 2:
                    bp_graph_node_ids.append(gene.gene_id)
        random.shuffle(bp_graph_node_ids)

        # Determine how many nodes will be removed based on the total node count
        number_of_nodes_to_rem = math.ceil(max_degree_of_mutation * node_count)

        # Remove nodes in loop until enough nodes are removed or until no node is left to be removed. When removing the
        # node, replace its incoming and outcoming connections with connections from each incoming node to each outgoing
        # node.
        rem_nodes_counter = 0
        while rem_nodes_counter < number_of_nodes_to_rem and len(bp_graph_node_ids) > 0:
            node_id_to_remove = bp_graph_node_ids.pop()
            node_to_remove = blueprint_graph[node_id_to_remove].node

            # Collect all gene ids with connections starting or ending in the chosen node, independent of if the node
            # is enabled or not to be removed later. Also collect all end nodes of the outgoing connections as well as
            # all start nodes of all incoming connections.
            conn_ids_to_remove = list()
            conn_replacement_start_nodes = list()
            conn_replacement_end_nodes = list()
            for gene in blueprint_graph.values():
                if isinstance(gene, CoDeepNEATBlueprintConn):
                    if gene.conn_start == node_to_remove:
                        conn_ids_to_remove.append(gene.gene_id)
                        if gene.enabled:
                            conn_replacement_end_nodes.append(gene.conn_end)
                    elif gene.conn_end == node_to_remove:
                        conn_ids_to_remove.append(gene.gene_id)
                        if gene.enabled:
                            conn_replacement_start_nodes.append(gene.conn_start)

            # Remove chosen node and all connections starting or ending in that node from blueprint graph
            del blueprint_graph[node_id_to_remove]
            parent_mutation['removed_genes'].append(node_id_to_remove)
            for id_to_remove in conn_ids_to_remove:
                del blueprint_graph[id_to_remove]
                parent_mutation['removed_genes'].append(id_to_remove)

            # Collect all current connections in blueprint graph to be checked against when creating new connections,
            # in case the connection already exists. This has be done in each iteration as those connections change
            # significantly for each round.
            bp_graph_conns = dict()
            for gene in blueprint_graph.values():
                if isinstance(gene, CoDeepNEATBlueprintConn):
                    bp_graph_conns[(gene.conn_start, gene.conn_end, gene.enabled)] = gene.gene_id

            # Recreate the connections of the removed node by connecting all start nodes of the incoming connections to
            # all end nodes of the outgoing connections. Only recreate the connection if the connection is not already
            # present or if the connection was disabled prior
            for new_start_node in conn_replacement_start_nodes:
                for new_end_node in conn_replacement_end_nodes:
                    # Check if enabled connection already present in bp graph. If so, skip creation and continue
                    if (new_start_node, new_end_node, True) in bp_graph_conns:
                        continue

                    # Check if a disabled variant of the connection in bp_graph. If so reenable it.
                    elif (new_start_node, new_end_node, False) in bp_graph_conns:
                        conn_id_to_reenable = bp_graph_conns[(new_start_node, new_end_node, False)]
                        blueprint_graph[conn_id_to_reenable].set_enabled(True)

                    # Check if a no variant of the connection to recreate is in the bp_graph. If so, create it.
                    else:  # (new_start_node, new_end_node, True) not in bp_graph_conns:
                        gene_id, gene = self.enc.create_blueprint_conn(conn_start=new_start_node,
                                                                       conn_end=new_end_node)
                        blueprint_graph[gene_id] = gene

        # Create and return the offspring blueprint with the edited blueprint graph having removed nodes which were
        # replaced by a full connection between all incoming and all outgoing nodes.
        return self.enc.create_blueprint(blueprint_graph=blueprint_graph,
                                         optimizer_factory=optimizer_factory,
                                         parent_mutation=parent_mutation)

    def _create_mutated_blueprint_node_spec(self, parent_blueprint, max_degree_of_mutation, mod_spec_extinct):
        """"""
        # Copy the parameters of the parent blueprint for the offspring
        blueprint_graph, optimizer_factory = parent_blueprint.copy_parameters()

        # Create the dict that keeps track of the way a blueprint has been mutated
        parent_mutation = {'parent_id': parent_blueprint.get_id(),
                           'mutation': 'node_spec',
                           'mutated_node_spec': dict()}

        # Determine if node species mutation even sensible and there exists more than 1 module species. Otherwise,
        # return the offspring identical to the parent blueprint
        available_mod_species = set(self.pop.mod_species.keys())
        if len(available_mod_species) >= 2:
            # Identify all non-Input nodes in the blueprint graph by gene ID as the species of those can be mutated.
            # Collect all gene ids with extinct node module species immediately
            bp_graph_node_ids = set()
            node_ids_to_change_species = set()
            for gene in blueprint_graph.values():
                if isinstance(gene, CoDeepNEATBlueprintNode) and gene.node != 1:
                    bp_graph_node_ids.add(gene.gene_id)
                    if gene.species in mod_spec_extinct:
                        node_ids_to_change_species.add(gene.gene_id)

            # Determine the amount and ids of node module species to change. Depending on if the forcibly changed node
            # module species aren't already a big enough mutation to satisfy the degree of mutation.
            number_of_node_species_to_change = math.ceil(max_degree_of_mutation * len(bp_graph_node_ids))
            if number_of_node_species_to_change > len(node_ids_to_change_species):
                additional_node_ids_count = number_of_node_species_to_change - len(node_ids_to_change_species)
                potential_additional_node_ids = bp_graph_node_ids.difference(node_ids_to_change_species)
                additional_node_ids = set(random.sample(potential_additional_node_ids, k=additional_node_ids_count))
                node_ids_to_change_species = node_ids_to_change_species.union(additional_node_ids)

            # Traverse through all randomly chosen node ids and change their module species randomly to another module
            # species, though not the original
            for node_id_to_change_species in node_ids_to_change_species:
                former_node_species = blueprint_graph[node_id_to_change_species].species
                parent_mutation['mutated_node_spec'][node_id_to_change_species] = former_node_species
                possible_new_node_species = tuple(available_mod_species - {former_node_species})
                blueprint_graph[node_id_to_change_species].species = random.choice(possible_new_node_species)

        # Create and return the offspring blueprint with the edited blueprint graph having mutated species
        return self.enc.create_blueprint(blueprint_graph=blueprint_graph,
                                         optimizer_factory=optimizer_factory,
                                         parent_mutation=parent_mutation)

    def _create_mutated_blueprint_optimizer(self, parent_blueprint):
        """"""
        # Copy the parameters of the parent blueprint for the offspring
        blueprint_graph, optimizer_factory = parent_blueprint.copy_parameters()
        parent_opt_params = optimizer_factory.get_parameters()

        # Create the dict that keeps track of the way a blueprint has been mutated
        parent_mutation = {'parent_id': parent_blueprint.get_id(),
                           'mutation': 'optimizer',
                           'mutated_params': parent_opt_params}

        # Randomly choose type of offspring optimizer and declare container collecting the specific parameters of
        # the offspring optimizer, setting only the chosen optimizer class
        offspring_optimizer_type = random.choice(self.available_optimizers)
        available_opt_params = self.available_opt_params[offspring_optimizer_type]
        offspring_opt_params = {'class_name': offspring_optimizer_type, 'config': dict()}

        if offspring_optimizer_type == parent_opt_params['class_name']:
            ## Mutation of the existing optimizers' parameters ##
            # Traverse each possible parameter option and determine a uniformly random value if its a categorical param
            # or try perturbing the the parent parameter if it is a sortable.
            for opt_param, opt_param_val_range in available_opt_params.items():
                # If the optimizer parameter is a categorical value choose randomly from the list
                if isinstance(opt_param_val_range, list):
                    offspring_opt_params['config'][opt_param] = random.choice(opt_param_val_range)
                # If the optimizer parameter is sortable, create a random value between the min and max values adhering
                # to the configured step
                elif isinstance(opt_param_val_range, dict):
                    if isinstance(opt_param_val_range['min'], int) \
                            and isinstance(opt_param_val_range['max'], int) \
                            and isinstance(opt_param_val_range['step'], int):
                        perturbed_param = int(np.random.normal(loc=parent_opt_params['config'][opt_param],
                                                               scale=opt_param_val_range['stddev']))
                        chosen_opt_param = round_with_step(perturbed_param,
                                                           opt_param_val_range['min'],
                                                           opt_param_val_range['max'],
                                                           opt_param_val_range['step'])
                    elif isinstance(opt_param_val_range['min'], float) \
                            and isinstance(opt_param_val_range['max'], float) \
                            and isinstance(opt_param_val_range['step'], float):
                        perturbed_param = np.random.normal(loc=parent_opt_params['config'][opt_param],
                                                           scale=opt_param_val_range['stddev'])
                        chosen_opt_param = round_with_step(perturbed_param,
                                                           opt_param_val_range['min'],
                                                           opt_param_val_range['max'],
                                                           opt_param_val_range['step'])
                    else:
                        raise NotImplementedError(f"Config parameter '{opt_param}' of the {offspring_optimizer_type} "
                                                  f"optimizer section is of type dict though the dict values are not "
                                                  f"of type int or float")
                    offspring_opt_params['config'][opt_param] = chosen_opt_param
                # If the optimizer parameter is a binary value it is specified as a float with the probablity of that
                # parameter being set to True
                elif isinstance(opt_param_val_range, float):
                    offspring_opt_params['config'][opt_param] = random.random() < opt_param_val_range
                else:
                    raise NotImplementedError(f"Config parameter '{opt_param}' of the {offspring_optimizer_type} "
                                              f"optimizer section is not one of the valid types of list, dict or float")

        else:
            ## Creation of a new optimizer with random parameters ##
            # Traverse each possible parameter option and determine a uniformly random value depending on if its a
            # categorical, sortable or boolean value
            for opt_param, opt_param_val_range in available_opt_params.items():
                # If the optimizer parameter is a categorical value choose randomly from the list
                if isinstance(opt_param_val_range, list):
                    offspring_opt_params['config'][opt_param] = random.choice(opt_param_val_range)
                # If the optimizer parameter is sortable, create a random value between the min and max values adhering
                # to the configured step
                elif isinstance(opt_param_val_range, dict):
                    if isinstance(opt_param_val_range['min'], int) \
                            and isinstance(opt_param_val_range['max'], int) \
                            and isinstance(opt_param_val_range['step'], int):
                        opt_param_random = random.randint(opt_param_val_range['min'],
                                                          opt_param_val_range['max'])
                        chosen_opt_param = round_with_step(opt_param_random,
                                                           opt_param_val_range['min'],
                                                           opt_param_val_range['max'],
                                                           opt_param_val_range['step'])
                    elif isinstance(opt_param_val_range['min'], float) \
                            and isinstance(opt_param_val_range['max'], float) \
                            and isinstance(opt_param_val_range['step'], float):
                        opt_param_random = random.uniform(opt_param_val_range['min'],
                                                          opt_param_val_range['max'])
                        chosen_opt_param = round_with_step(opt_param_random,
                                                           opt_param_val_range['min'],
                                                           opt_param_val_range['max'],
                                                           opt_param_val_range['step'])
                    else:
                        raise NotImplementedError(f"Config parameter '{opt_param}' of the {offspring_optimizer_type} "
                                                  f"optimizer section is of type dict though the dict values are not "
                                                  f"of type int or float")
                    offspring_opt_params['config'][opt_param] = chosen_opt_param
                # If the optimizer parameter is a binary value it is specified as a float with the probablity of that
                # parameter being set to True
                elif isinstance(opt_param_val_range, float):
                    offspring_opt_params['config'][opt_param] = random.random() < opt_param_val_range
                else:
                    raise NotImplementedError(f"Config parameter '{opt_param}' of the {offspring_optimizer_type} "
                                              f"optimizer section is not one of the valid types of list, dict or float")

        # Create new optimizer through encoding, having either the parent perturbed offspring parameters or randomly
        # new created parameters
        optimizer_factory = self.enc.create_optimizer_factory(optimizer_parameters=offspring_opt_params)

        # Create and return the offspring blueprint with identical blueprint graph and modified optimizer_factory
        return self.enc.create_blueprint(blueprint_graph=blueprint_graph,
                                         optimizer_factory=optimizer_factory,
                                         parent_mutation=parent_mutation)

    def _create_crossed_over_blueprint(self, parent_bp_1, parent_bp_2):
        """"""
        # Copy the parameters of both parent blueprints for the offspring
        bp_graph_1, opt_factory_1 = parent_bp_1.copy_parameters()
        bp_graph_2, opt_factory_2 = parent_bp_2.copy_parameters()

        # Create the dict that keeps track of the way a blueprint has been mutated
        parent_mutation = {'parent_id': (parent_bp_1.get_id(), parent_bp_2.get_id()),
                           'mutation': 'crossover',
                           'gene_parent': dict(),
                           'optimizer_parent': None}

        ### Crossover ###
        # Create quickly searchable sets of gene ids to know about the overlap of genes in both blueprint graphs
        bp_graph_1_ids = set(bp_graph_1.keys())
        bp_graph_2_ids = set(bp_graph_2.keys())
        all_bp_graph_ids = bp_graph_1_ids.union(bp_graph_2_ids)

        # Create offspring blueprint graph by traversing all blueprint graph ids and copying over all genes that are
        # exclusive to either blueprint graph and randomly choosing the gene to copy over that is present in both graphs
        offspring_bp_graph = dict()
        for gene_id in all_bp_graph_ids:
            if gene_id in bp_graph_1_ids and gene_id in bp_graph_2_ids:
                if random.choice((True, False)):
                    offspring_bp_graph[gene_id] = bp_graph_1[gene_id]
                    parent_mutation['gene_parent'][gene_id] = parent_bp_1.get_id()
                else:
                    offspring_bp_graph[gene_id] = bp_graph_2[gene_id]
                    parent_mutation['gene_parent'][gene_id] = parent_bp_2.get_id()
            elif gene_id in bp_graph_1_ids:
                offspring_bp_graph[gene_id] = bp_graph_1[gene_id]
                parent_mutation['gene_parent'][gene_id] = parent_bp_1.get_id()
            else:  # if gene_id in bp_graph_2_ids
                offspring_bp_graph[gene_id] = bp_graph_2[gene_id]
                parent_mutation['gene_parent'][gene_id] = parent_bp_2.get_id()

        ### Recurrent Correction ####
        # Disable recurrent connections created in crossover, as CoDeepNEAT currently only supports feedforward
        # topologies. First determine node dependencies as required for a circular dependency check
        node_deps = dict()
        for gene in offspring_bp_graph.values():
            if isinstance(gene, CoDeepNEATBlueprintConn):
                if gene.conn_end in node_deps:
                    node_deps[gene.conn_end].add(gene.conn_start)
                else:
                    node_deps[gene.conn_end] = {gene.conn_start}
        node_deps[1] = set()

        # Perform a circular dependency check by iteratively removing the dependencyless nodes. If a circular dependency
        # is detected, determine the nodes for this circular dependency and adjust the offspring bp graph
        circular_dep_flag = False
        orig_node_deps = node_deps.copy()
        while True:
            # Find all nodes in graph having no dependencies in current iteration
            dependencyless = set()
            for node, dep in node_deps.items():
                if len(dep) == 0:
                    dependencyless.add(node)

            # If no dependencyless node was found but there are still node dependencies present, then the graph has a
            # circular dependency. Find this circular dependent connection and remove it
            if not dependencyless and node_deps:
                circular_dep_flag = True
                circular_dependent_conn = None
                possibly_circ_dep_conns = {k: v for k, v in node_deps.items() if v != orig_node_deps[k]}

                # Determine potentially circular dependent connected nodes and determine which connections are involved
                # in the circular dependency by creating for each connection a dependency chain and checking if
                # eventually a node is dependent on itself
                possibly_circ_dep_conn_ends = sorted(possibly_circ_dep_conns.keys(), reverse=True)
                for possible_conn_end in possibly_circ_dep_conn_ends:
                    for possible_conn_start in possibly_circ_dep_conns[possible_conn_end]:
                        dep_chain = {possible_conn_start}
                        dep_chain_checked_paths = list()
                        while dep_chain:
                            orig_dep_chain = dep_chain.copy()
                            for i in orig_dep_chain:
                                if i == possible_conn_end:
                                    circular_dependent_conn = (possible_conn_start, possible_conn_end)
                                    break
                                dep_chain.remove(i)
                                if i not in dep_chain_checked_paths:
                                    dep_chain = dep_chain.union(node_deps[i])
                                    dep_chain_checked_paths.append(i)
                            if circular_dependent_conn is not None:
                                break
                        if circular_dependent_conn is not None:
                            break
                    if circular_dependent_conn is not None:
                        break

                # Find gene id belonging to the determined circular dependent connection and remove it
                gene_id_to_remove = None
                for gene in offspring_bp_graph.values():
                    if isinstance(gene, CoDeepNEATBlueprintConn) \
                            and gene.conn_start == circular_dependent_conn[0] \
                            and gene.conn_end == circular_dependent_conn[1]:
                        gene_id_to_remove = gene.gene_id
                        break
                del offspring_bp_graph[gene_id_to_remove]
                del parent_mutation['gene_parent'][gene_id_to_remove]

                # Adjust node dependencies and and restart loop in case there are more than 1 circular dependencies
                node_deps = orig_node_deps
                node_deps[circular_dependent_conn[1]].remove(circular_dependent_conn[0])
                orig_node_deps = node_deps.copy()
                continue

            elif not dependencyless and not node_deps:
                # No circular dependency occuring. Leave loop
                break

            # Remove keys with empty dependencies and remove all nodes that are considered dependencyless from the
            # dependencies of other nodes in order to create next iteration
            for node in dependencyless:
                del node_deps[node]
            for node, dep in node_deps.items():
                node_deps[node] = dep - dependencyless

        # Check for orphaned nodes that don't have any incoming or outgoing connections as they could have possibly
        # been removed when correcting for circular dependencies. Add incoming conns from the start node or add outgoing
        # conns to the end node
        if circular_dep_flag:
            present_nodes = list()
            outgoing_conns = list()
            incoming_conns = list()
            for gene in offspring_bp_graph.values():
                if isinstance(gene, CoDeepNEATBlueprintNode):
                    present_nodes.append(gene.node)
                if isinstance(gene, CoDeepNEATBlueprintConn) and gene.enabled:
                    incoming_conns.append(gene.conn_end)
                    outgoing_conns.append(gene.conn_start)

            missing_conns = list()
            for node in present_nodes:
                if node != 1 and node not in incoming_conns:
                    # Node is missing an incoming connection. Create connection from input node.
                    missing_conns.append((1, node))
                if node != 2 and node not in outgoing_conns:
                    # Node is missing an outgoing connection. Create connection to output node.
                    missing_conns.append((node, 2))

            # Create missing connections and add them to the offspring blueprint graph
            for missing_conn in missing_conns:
                gene_id, gene = self.enc.create_blueprint_conn(conn_start=missing_conn[0],
                                                               conn_end=missing_conn[1])
                offspring_bp_graph[gene_id] = gene
                parent_mutation['gene_parent'][gene_id] = 'orphaned node correction'

        ### Optimizer and Blueprint Creation ###
        # For the optimizer factory choose the one from the fitter parent blueprint
        if parent_bp_1.get_fitness() > parent_bp_2.get_fitness():
            offspring_opt_factory = opt_factory_1
            parent_mutation['optimizer_parent'] = parent_bp_1.get_id()
        else:
            offspring_opt_factory = opt_factory_2
            parent_mutation['optimizer_parent'] = parent_bp_2.get_id()

        # Create and return the offspring blueprint with crossed over blueprint graph and optimizer_factory of the
        # fitter parent
        return self.enc.create_blueprint(blueprint_graph=offspring_bp_graph,
                                         optimizer_factory=offspring_opt_factory,
                                         parent_mutation=parent_mutation)
