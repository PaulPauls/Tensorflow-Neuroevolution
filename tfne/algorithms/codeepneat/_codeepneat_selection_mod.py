import math
from typing import Union


class CoDeepNEATSelectionMOD:
    def _select_modules_basic(self) -> ({Union[int, str]: int}, {int: int}, {int}):
        """"""
        ### Generational Parent Determination ###
        # Determine potential parents of the module species for offspring creation. Modules are ordered by their fitness
        # and the top x percent of those modules (as dictated via the reproduction threshold parameter) are chosen as
        # generational parents. The species elite modules (best members) are added as potential parents
        mod_spec_parents = dict()
        for spec_id, spec_mod_ids in self.pop.mod_species.items():
            # Sort module ids in species according to their fitness
            spec_mod_ids_sorted = sorted(spec_mod_ids, key=lambda x: self.pop.modules[x].get_fitness())

            # Determine the species elite as the top x members
            spec_elites = set(spec_mod_ids_sorted[-self.mod_spec_mod_elitism:])

            # Determine the species parents as those clearing the reproduction threshold, plus the species elites
            reprod_threshold_index = math.ceil(len(spec_mod_ids) * self.mod_spec_reprod_thres)
            spec_parents = set(spec_mod_ids_sorted[reprod_threshold_index:])
            spec_parents = spec_parents.union(spec_elites)

            # Remove non elite modules from the species list, as they are not part of the species anymore. Remove non
            # parental modules from the module container as there is no use of those modules anymore.
            mod_ids_non_elite = set(spec_mod_ids) - spec_elites
            mod_ids_non_parental = set(spec_mod_ids) - spec_parents
            for mod_id in mod_ids_non_elite:
                self.pop.mod_species[spec_id].remove(mod_id)
            for mod_id in mod_ids_non_parental:
                del self.pop.modules[mod_id]

            # Cast potential parents to tuple, as randomly chosen from
            mod_spec_parents[spec_id] = tuple(spec_parents)

        #### Offspring Size Calculation ####
        # Determine the amount of offspring for each species. Each species is assigned offspring according to its
        # species share of the total fitness, though minimum offspring constraints are considered. Preprocess by
        # determining the sum of all average fitness and creating a module species order
        total_avg_fitness = 0
        for fitness_history in self.pop.mod_species_fitness_history.values():
            total_avg_fitness += fitness_history[self.pop.generation_counter]
        mod_species_ordered = sorted(self.pop.mod_species.keys(),
                                     key=lambda x: self.pop.mod_species_fitness_history[x][self.pop.generation_counter])

        # Work through each species in order (from least to most fit) and determine the intended size as the species
        # fitness share of the total fitness of the remaining species, applied to the remaining population slots.
        # Assign offspring under the consideration of the minimal offspring constraint and then decrease the total
        # fitness and the remaining population slots.
        mod_spec_offspring = dict()
        available_mod_pop = self.mod_pop_size
        for spec_id in mod_species_ordered:
            spec_fitness = self.pop.mod_species_fitness_history[spec_id][self.pop.generation_counter]
            spec_fitness_share = spec_fitness / total_avg_fitness
            spec_intended_size = int(round(spec_fitness_share * available_mod_pop))

            if len(self.pop.mod_species[spec_id]) + self.mod_spec_min_offspring > spec_intended_size:
                mod_spec_offspring[spec_id] = self.mod_spec_min_offspring
                available_mod_pop -= len(self.pop.mod_species[spec_id]) + self.mod_spec_min_offspring
            else:
                mod_spec_offspring[spec_id] = spec_intended_size - len(self.pop.mod_species[spec_id])
                available_mod_pop -= spec_intended_size
            total_avg_fitness -= spec_fitness

        # Return
        # mod_spec_offspring {int: int} associating species id with amount of offspring
        # mod_spec_parents {int: [int]} associating species id with list of potential parent ids for species
        # Return empty list for extinct species, as no species extinction in basic speciation
        mod_spec_extinct = list()
        return mod_spec_offspring, mod_spec_parents, mod_spec_extinct

    def _select_modules_param_distance_fixed(self) -> ({Union[int, str]: int}, {int: int}, {int}):
        """"""
        ### Species Extinction ###
        # Determine if species can be considered for extinction. Critera: Species existed long enough; species can be
        # removed according to species elitism; species is not the last of its module type. Then determine if species is
        # stagnating for the recent config specified time period (meaning that it had not improved at any time in the
        # recent time period).
        # Preprocess current species by listing the frequency of module types as to not remove the last species of a
        # unique module type.
        spec_type_frequency = dict()
        for mod_id in self.pop.mod_species_repr.values():
            spec_mod_type = self.pop.modules[mod_id].get_module_type()
            if spec_mod_type in spec_type_frequency:
                spec_type_frequency[spec_mod_type] += 1
            else:
                spec_type_frequency[spec_mod_type] = 1

        # Order species according to their fitness in order to remove low performing species first
        mod_species_ordered = sorted(self.pop.mod_species.keys(),
                                     key=lambda x: self.pop.mod_species_fitness_history[x][self.pop.generation_counter])

        # Traverse ordered species list. Keep track of extinct species and the total fitness achieved by those extinct
        # species as relevant for later calculation of reinitialized offspring, if option activated.
        extinct_fitness = 0
        mod_spec_extinct = set()
        for spec_id in mod_species_ordered:
            # Don't consider species for extinction if it hasn't existed long enough
            if len(self.pop.mod_species_fitness_history[spec_id]) < self.mod_spec_max_stagnation + 1:
                continue
            # Don't consider species for extinction if species elitism doesn't allow removal of further species
            if len(self.pop.mod_species) <= self.mod_spec_species_elitism:
                continue
            # Don't consider species for extinction if it is the last of its module type and species elitism is set to
            # a value higher than all possible module types.
            spec_mod_type = self.pop.modules[self.pop.mod_species_repr[spec_id]].get_module_type()
            if spec_type_frequency[spec_mod_type] == 1 and self.mod_spec_species_elitism >= len(self.available_modules):
                continue

            # Consider species for extinction and determine if it has been stagnating by checking if the distant avg
            # fitness is higher than all recent avg fitnesses
            distant_generation = self.pop.generation_counter - self.mod_spec_max_stagnation
            distant_avg_fitness = self.pop.mod_species_fitness_history[spec_id][distant_generation]
            recent_fitness = list()
            for i in range(self.mod_spec_max_stagnation):
                recent_fitness.append(self.pop.mod_species_fitness_history[spec_id][self.pop.generation_counter - i])
            if distant_avg_fitness >= max(recent_fitness):
                # Species is stagnating. Flag species as extinct, keep track of its fitness and then remove it from the
                # population
                mod_spec_extinct.add(spec_id)
                extinct_fitness += self.pop.mod_species_fitness_history[spec_id][self.pop.generation_counter]
                spec_type_frequency[spec_mod_type] -= 1
                for mod_id in self.pop.mod_species[spec_id]:
                    del self.pop.modules[mod_id]
                del self.pop.mod_species[spec_id]
                del self.pop.mod_species_repr[spec_id]
                del self.pop.mod_species_fitness_history[spec_id]

        ### Rebase Species Representative ###
        # If Rebase representative config flag set to true, rechoose the representative of each species as the best
        # module of the species that also holds the minimum set distance ('mod_spec_distance') to all other species
        # representatives. Begin the rebasing of species representatives from the oldest to the newest species.
        if self.mod_spec_rebase_repr:
            all_spec_repr_ids = set(self.pop.mod_species_repr.values())
            for spec_id, spec_repr_id in self.pop.mod_species_repr.items():
                # Determine the module ids of all other species representatives and create a sorted list of the modules
                # in the current species according to their fitness
                other_spec_repr_ids = all_spec_repr_ids - {spec_repr_id}

                # Traverse each module id in the sorted module id list beginning with the best. Determine the distance
                # to other species representative module ids and if the distance to all other species representatives is
                # higher than the specified minimum distance for a new species, set the module as the new
                # representative.
                spec_mod_ids_sorted = sorted(self.pop.mod_species[spec_id],
                                             key=lambda x: self.pop.modules[x].get_fitness(),
                                             reverse=True)
                for mod_id in spec_mod_ids_sorted:
                    if mod_id == spec_repr_id:
                        # Best species module already representative. Abort search.
                        break
                    module = self.pop.modules[mod_id]
                    distance_to_other_spec_repr = [module.get_distance(self.pop.modules[other_mod_id])
                                                   for other_mod_id in other_spec_repr_ids]
                    if all(distance >= self.mod_spec_distance for distance in distance_to_other_spec_repr):
                        # New best species representative found. Set as representative and abort search
                        self.pop.mod_species_repr[spec_id] = mod_id
                        break

        ### Generational Parent Determination ###
        # Determine potential parents of the module species for offspring creation. Modules are ordered by their fitness
        # and the top x percent of those modules (as dictated via the reproduction threshold parameter) are chosen as
        # generational parents. The species elite modules (best members and representative) are added as potential
        # parents, even if the representative does not make the cut according to the reproduction threshold.
        mod_spec_parents = dict()
        for spec_id, spec_mod_ids in self.pop.mod_species.items():
            # Sort module ids in species according to their fitness
            spec_mod_ids_sorted = sorted(spec_mod_ids, key=lambda x: self.pop.modules[x].get_fitness())

            # Determine the species elite as the top x members and the species representative
            spec_elites = set(spec_mod_ids_sorted[-self.mod_spec_mod_elitism:])
            spec_elites.add(self.pop.mod_species_repr[spec_id])

            # Determine the species parents as those clearing the reproduction threshold, plus the species elites
            reprod_threshold_index = math.ceil(len(spec_mod_ids) * self.mod_spec_reprod_thres)
            spec_parents = set(spec_mod_ids_sorted[reprod_threshold_index:])
            spec_parents = spec_parents.union(spec_elites)

            # Remove non elite modules from the species list, as they are not part of the species anymore. Remove non
            # parental modules from the module container as there is no use of those modules anymore.
            mod_ids_non_elite = set(spec_mod_ids) - spec_elites
            mod_ids_non_parental = set(spec_mod_ids) - spec_parents
            for mod_id in mod_ids_non_elite:
                self.pop.mod_species[spec_id].remove(mod_id)
            for mod_id in mod_ids_non_parental:
                del self.pop.modules[mod_id]

            # Cast potential parents to tuple, as randomly chosen from
            mod_spec_parents[spec_id] = tuple(spec_parents)

        #### Offspring Size Calculation ####
        # Determine the amount of offspring for each species as well as the amount of reinitialized modules, in case
        # this option is activated. Each species is assigned offspring according to its species share of the total
        # fitness, though minimum offspring constraints are considered. Preprocess by determining the sum of all
        # average fitness and removing the extinct species from the species order
        total_avg_fitness = 0
        for fitness_history in self.pop.mod_species_fitness_history.values():
            total_avg_fitness += fitness_history[self.pop.generation_counter]
        for spec_id in mod_spec_extinct:
            mod_species_ordered.remove(spec_id)

        # Determine the amount of offspring to be reinitialized as the fitness share of the total fitness by the extinct
        # species
        mod_spec_offspring = dict()
        available_mod_pop = self.mod_pop_size
        if self.mod_spec_reinit_extinct and extinct_fitness > 0:
            extinct_fitness_share = extinct_fitness / (total_avg_fitness + extinct_fitness)
            reinit_offspring = int(extinct_fitness_share * available_mod_pop)
            mod_spec_offspring['reinit'] = reinit_offspring
            available_mod_pop -= reinit_offspring

        # Work through each species in order (from least to most fit) and determine the intended size as the species
        # fitness share of the total fitness of the remaining species, applied to the remaining population slots.
        # Assign offspring under the consideration of the minimal offspring constraint and then decrease the total
        # fitness and the remaining population slots.
        for spec_id in mod_species_ordered:
            spec_fitness = self.pop.mod_species_fitness_history[spec_id][self.pop.generation_counter]
            spec_fitness_share = spec_fitness / total_avg_fitness
            spec_intended_size = int(round(spec_fitness_share * available_mod_pop))

            if len(self.pop.mod_species[spec_id]) + self.mod_spec_min_offspring > spec_intended_size:
                mod_spec_offspring[spec_id] = self.mod_spec_min_offspring
                available_mod_pop -= len(self.pop.mod_species[spec_id]) + self.mod_spec_min_offspring
            else:
                mod_spec_offspring[spec_id] = spec_intended_size - len(self.pop.mod_species[spec_id])
                available_mod_pop -= spec_intended_size
            total_avg_fitness -= spec_fitness

        # Return
        # mod_spec_offspring {int: int} associating species id with amount of offspring
        # mod_spec_parents {int: [int]} associating species id with list of potential parent ids for species
        # mod_spec_extinct {int} listing all modules species that went extinct in this generation
        return mod_spec_offspring, mod_spec_parents, mod_spec_extinct

    def _select_modules_param_distance_dynamic(self) -> ({Union[int, str]: int}, {int: int}, {int}):
        """"""
        # selection process identical for both variants of module speciation
        return self._select_modules_param_distance_fixed()
