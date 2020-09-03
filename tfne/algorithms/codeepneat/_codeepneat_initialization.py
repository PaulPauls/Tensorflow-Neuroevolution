import random

from tfne.encodings.codeepneat import CoDeepNEATBlueprint
from tfne.helper_functions import round_with_step


class CoDeepNEATInitialization:
    def _create_initial_blueprint(self, initial_node_species) -> (int, CoDeepNEATBlueprint):
        """"""
        # Create the dict that keeps track of the way a blueprint has been mutated or created
        parent_mutation = {'parent_id': None,
                           'mutation': 'init'}

        # Create a minimal blueprint graph with node 1 being the input node (having no species) and node 2 being the
        # random initial node species
        blueprint_graph = dict()
        gene_id, gene = self.enc.create_blueprint_node(node=1, species=None)
        blueprint_graph[gene_id] = gene
        gene_id, gene = self.enc.create_blueprint_node(node=2, species=initial_node_species)
        blueprint_graph[gene_id] = gene
        gene_id, gene = self.enc.create_blueprint_conn(conn_start=1, conn_end=2)
        blueprint_graph[gene_id] = gene

        # Randomly choose an optimizer from the available optimizers and create the parameter config dict of it
        chosen_optimizer = random.choice(self.available_optimizers)
        available_optimizer_params = self.available_opt_params[chosen_optimizer]

        # Declare container collecting the specific parameters of the optimizer to be created, setting the just chosen
        # optimizer class
        chosen_optimizer_params = {'class_name': chosen_optimizer, 'config': dict()}

        # Traverse each possible parameter option and determine a uniformly random value depending on if its a
        # categorical, sortable or boolean value
        for opt_param, opt_param_val_range in available_optimizer_params.items():
            # If the optimizer parameter is a categorical value choose randomly from the list
            if isinstance(opt_param_val_range, list):
                chosen_optimizer_params['config'][opt_param] = random.choice(opt_param_val_range)
            # If the optimizer parameter is sortable, create a random value between the min and max values adhering
            # to the configured step
            elif isinstance(opt_param_val_range, dict):
                if isinstance(opt_param_val_range['min'], int) and isinstance(opt_param_val_range['max'], int) \
                        and isinstance(opt_param_val_range['step'], int):
                    opt_param_random = random.randint(opt_param_val_range['min'],
                                                      opt_param_val_range['max'])
                    chosen_opt_param = round_with_step(opt_param_random,
                                                       opt_param_val_range['min'],
                                                       opt_param_val_range['max'],
                                                       opt_param_val_range['step'])
                elif isinstance(opt_param_val_range['min'], float) and isinstance(opt_param_val_range['max'], float) \
                        and isinstance(opt_param_val_range['step'], float):
                    opt_param_random = random.uniform(opt_param_val_range['min'],
                                                      opt_param_val_range['max'])
                    chosen_opt_param = round_with_step(opt_param_random,
                                                       opt_param_val_range['min'],
                                                       opt_param_val_range['max'],
                                                       opt_param_val_range['step'])
                else:
                    raise NotImplementedError(f"Config parameter '{opt_param}' of the {chosen_optimizer} optimizer "
                                              f"section is of type dict though the dict values are not of type int or "
                                              f"float")
                chosen_optimizer_params['config'][opt_param] = chosen_opt_param
            # If the optimizer parameter is a binary value it is specified as a float with the probablity of that
            # parameter being set to True
            elif isinstance(opt_param_val_range, float):
                chosen_optimizer_params['config'][opt_param] = random.random() < opt_param_val_range
            else:
                raise NotImplementedError(f"Config parameter '{opt_param}' of the {chosen_optimizer} optimizer section "
                                          f"is not one of the valid types of list, dict or float")

        # Create new optimizer through encoding
        optimizer_factory = self.enc.create_optimizer_factory(optimizer_parameters=chosen_optimizer_params)

        # Create just defined initial blueprint through encoding
        return self.enc.create_blueprint(blueprint_graph=blueprint_graph,
                                         optimizer_factory=optimizer_factory,
                                         parent_mutation=parent_mutation)
