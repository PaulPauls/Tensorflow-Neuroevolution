import random


class CoDeepNEATEvolutionMOD:
    def _evolve_modules(self, mod_spec_offspring, mod_spec_parents) -> [int]:
        """"""
        # Create container for new modules that will be speciated in a later function
        new_module_ids = list()

        #### Evolve Modules ####
        # Traverse through each species and create according amount of offspring as determined prior during selection
        for spec_id, species_offspring in mod_spec_offspring.items():
            if spec_id == 'reinit':
                continue

            for _ in range(species_offspring):
                # Choose randomly between mutation or crossover of module
                if random.random() < self.mod_mutation_prob:
                    ## Create new module through mutation ##
                    # Determine random maximum degree of mutation > 0 and randomly choose a parent module from the
                    # remaining modules of the current species. Create a mutation by letting the module internal
                    # function take care of this.
                    max_degree_of_mutation = random.uniform(1e-323, self.mod_max_mutation)
                    parent_module = self.pop.modules[random.choice(mod_spec_parents[spec_id])]
                    new_mod_id, new_mod = self.enc.create_mutated_module(parent_module, max_degree_of_mutation)

                else:  # random.random() < self.mod_mutation_prob + self.mod_crossover_prob
                    ## Create new module through crossover ##
                    # Determine if species has at least 2 modules as required for crossover
                    if len(mod_spec_parents[spec_id]) >= 2:
                        # Determine the 2 parent modules used for crossover
                        parent_module_1_id, parent_module_2_id = random.sample(mod_spec_parents[spec_id], k=2)
                        parent_module_1 = self.pop.modules[parent_module_1_id]
                        parent_module_2 = self.pop.modules[parent_module_2_id]

                        # Randomly determine the maximum degree of mutation > 0 and let the modules internal function
                        # create a crossover
                        max_degree_of_mutation = random.uniform(1e-323, self.mod_max_mutation)
                        new_mod_id, new_mod = self.enc.create_crossover_module(parent_module_1,
                                                                               parent_module_2,
                                                                               max_degree_of_mutation)

                    else:
                        # As species does not have enough modules for crossover, perform a mutation on the remaining
                        # module
                        max_degree_of_mutation = random.uniform(1e-323, self.mod_max_mutation)
                        parent_module = self.pop.modules[random.choice(mod_spec_parents[spec_id])]
                        new_mod_id, new_mod = self.enc.create_mutated_module(parent_module, max_degree_of_mutation)

                # Add newly created module to the module container and to the list of modules that have to be speciated
                self.pop.modules[new_mod_id] = new_mod
                new_module_ids.append(new_mod_id)

        #### Reinitialize Modules ####
        if 'reinit' in mod_spec_offspring:
            # Initialize predetermined number of new modules as species went extinct and reinitialization is activated
            for i in range(mod_spec_offspring['reinit']):
                # Decide on for which species a new module is added (uniformly distributed)
                chosen_species = i % len(self.available_modules)

                # Determine type and the associated config parameters of chosen species and initialize a module with it
                mod_type = self.available_modules[chosen_species]
                mod_config_params = self.available_mod_params[mod_type]
                new_mod_id, new_mod = self.enc.create_initial_module(mod_type=mod_type,
                                                                     config_params=mod_config_params)

                # Add newly created module to the module container and to the list of modules that have to be speciated
                self.pop.modules[new_mod_id] = new_mod
                new_module_ids.append(new_mod_id)

        # Return the list of new module ids for later speciation
        return new_module_ids
