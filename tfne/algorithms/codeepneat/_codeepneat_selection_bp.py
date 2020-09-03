import math
from typing import Union


class CoDeepNEATSelectionBP:
    def _select_blueprints_basic(self) -> ({Union[int, str]: int}, {int: int}):
        """"""
        ### Generational Parent Determination ###
        # Determine potential parents of the blueprint species for offspring creation. Blueprints are ordered by their
        # fitness and the top x percent of those blueprints (as dictated via the reproduction threshold parameter) are
        # chosen as generational parents. The species elite blueprints (best members) are added as potential parents.
        # First sort blueprint ids in species according to their fitness
        spec_bp_ids = self.pop.bp_species[1]
        spec_bp_ids_sorted = sorted(spec_bp_ids, key=lambda x: self.pop.blueprints[x].get_fitness())

        # Determine the species elite as the top x members
        spec_elites = set(spec_bp_ids_sorted[-self.bp_spec_bp_elitism:])

        # Determine the species parents as those clearing the reproduction threshold, plus the species elites
        reprod_threshold_index = math.ceil(len(spec_bp_ids) * self.bp_spec_reprod_thres)
        spec_parents = set(spec_bp_ids_sorted[reprod_threshold_index:])
        spec_parents = spec_parents.union(spec_elites)

        # Remove non elite blueprints from the species list, as they are not part of the species anymore. Remove non
        # parental blueprints from the blueprint container as there is no use of those blueprints anymore.
        bp_ids_non_elite = set(spec_bp_ids) - spec_elites
        bp_ids_non_parental = set(spec_bp_ids) - spec_parents
        for bp_id in bp_ids_non_elite:
            self.pop.bp_species[1].remove(bp_id)
        for bp_id in bp_ids_non_parental:
            del self.pop.blueprints[bp_id]

        # Cast potential parents to list, as potentially adjusted in BP evolution
        bp_spec_parents = {1: list(spec_parents)}

        #### Offspring Size Calculation ####
        # Determine the amount of offspring for the blueprint species as the difference between the intended population
        # size and the amount of elite blueprints carried over to the next generation.
        bp_spec_offspring = {1: self.bp_pop_size - len(self.pop.bp_species[1])}

        # Return
        # bp_spec_offspring {int: int} associating species id with amount of offspring
        # bp_spec_parents {int: [int]} associating species id with list of potential parent ids for species
        return bp_spec_offspring, bp_spec_parents

    def _select_blueprints_gene_overlap_fixed(self) -> ({Union[int, str]: int}, {int: int}):
        """"""
        ### Species Extinction ###
        # Determine if species can be considered for extinction. Critera: Species existed long enough; species can be
        # removed according to species elitism; Then determine if species is stagnating for the recent config specified
        # time period (meaning that it had not improved at any time in the recent time period).
        # Order species according to their fitness in order to remove low performing species first
        bp_species_ordered = sorted(self.pop.bp_species.keys(),
                                    key=lambda x: self.pop.bp_species_fitness_history[x][self.pop.generation_counter])

        # Traverse ordered species list. Keep track of extinct species and the total fitness achieved by those extinct
        # species as relevant for later calculation of reinitialized offspring, if option activated.
        extinct_fitness = 0
        bp_spec_extinct = set()
        for spec_id in bp_species_ordered:
            # Don't consider species for extinction if it hasn't existed long enough
            if len(self.pop.bp_species_fitness_history[spec_id]) < self.bp_spec_max_stagnation + 1:
                continue
            # Don't consider species for extinction if species elitism doesn't allow removal of further species
            if len(self.pop.bp_species) <= self.bp_spec_species_elitism:
                continue

            # Consider species for extinction and determine if it has been stagnating by checking if the distant avg
            # fitness is higher than all recent avg fitnesses
            distant_generation = self.pop.generation_counter - self.bp_spec_max_stagnation
            distant_avg_fitness = self.pop.bp_species_fitness_history[spec_id][distant_generation]
            recent_fitness = list()
            for i in range(self.bp_spec_max_stagnation):
                recent_fitness.append(self.pop.bp_species_fitness_history[spec_id][self.pop.generation_counter - i])
            if distant_avg_fitness >= max(recent_fitness):
                # Species is stagnating. Flag species as extinct, keep track of its fitness and then remove it from the
                # population
                bp_spec_extinct.add(spec_id)
                extinct_fitness += self.pop.bp_species_fitness_history[spec_id][self.pop.generation_counter]
                for bp_id in self.pop.bp_species[spec_id]:
                    del self.pop.blueprints[bp_id]
                del self.pop.bp_species[spec_id]
                del self.pop.bp_species_repr[spec_id]
                del self.pop.bp_species_fitness_history[spec_id]

        ### Rebase Species Representative ###
        # If Rebase representative config flag set to true, rechoose the representative of each species as the best
        # blueprint of the species that also holds the minimum set distance ('bp_spec_distance') to all other species
        # representatives. Begin the rebasing of species representatives from the oldest to the newest species.
        if self.bp_spec_rebase_repr:
            all_spec_repr_ids = set(self.pop.bp_species_repr.values())
            for spec_id, spec_repr_id in self.pop.bp_species_repr.items():
                # Determine the blueprint ids of all other species representatives and create a sorted list of the
                # blueprints in the current species according to their fitness
                other_spec_repr_ids = all_spec_repr_ids - {spec_repr_id}

                # Traverse each blueprint id in the sorted blueprint id list beginning with the best. Determine the
                # distance to other species representative blueprint ids and if the distance to all other species
                # representatives is higher than the specified minimum distance for a new species, set the blueprint
                # as the new representative.
                spec_bp_ids_sorted = sorted(self.pop.bp_species[spec_id],
                                            key=lambda x: self.pop.blueprints[x].get_fitness(),
                                            reverse=True)
                for bp_id in spec_bp_ids_sorted:
                    if bp_id == spec_repr_id:
                        # Best species blueprint already representative. Abort search.
                        break
                    blueprint = self.pop.blueprints[bp_id]
                    distance_to_other_spec_repr = [blueprint.calculate_gene_distance(self.pop.blueprints[other_bp_id])
                                                   for other_bp_id in other_spec_repr_ids]
                    if all(distance >= self.bp_spec_distance for distance in distance_to_other_spec_repr):
                        # New best species representative found. Set as representative and abort search
                        self.pop.bp_species_repr[spec_id] = bp_id
                        break

        ### Generational Parent Determination ###
        # Determine potential parents of the blueprint species for offspring creation. Blueprints are ordered by their
        # fitness and the top x percent of those blueprints (as dictated via the reproduction threshold parameter) are
        # chosen as generational parents. The species elite blueprints (best members and representative) are added as
        # potential parents, even if the representative does not make the cut according to the reproduction threshold.
        bp_spec_parents = dict()
        for spec_id, spec_bp_ids in self.pop.bp_species.items():
            # Sort blueprint ids in species according to their fitness
            spec_bp_ids_sorted = sorted(spec_bp_ids, key=lambda x: self.pop.blueprints[x].get_fitness())

            # Determine the species elite as the top x members and the species representative
            spec_elites = set(spec_bp_ids_sorted[-self.bp_spec_bp_elitism:])
            spec_elites.add(self.pop.bp_species_repr[spec_id])

            # Determine the species parents as those clearing the reproduction threshold, plus the species elites
            reprod_threshold_index = math.ceil(len(spec_bp_ids) * self.bp_spec_reprod_thres)
            spec_parents = set(spec_bp_ids_sorted[reprod_threshold_index:])
            spec_parents = spec_parents.union(spec_elites)

            # Remove non elite blueprints from the species list, as they are not part of the species anymore. Remove non
            # parental blueprints from the blueprint container as there is no use of those blueprints anymore.
            bp_ids_non_elite = set(spec_bp_ids) - spec_elites
            bp_ids_non_parental = set(spec_bp_ids) - spec_parents
            for bp_id in bp_ids_non_elite:
                self.pop.bp_species[spec_id].remove(bp_id)
            for bp_id in bp_ids_non_parental:
                del self.pop.blueprints[bp_id]

            # Cast potential parents to list, as potentially adjusted in BP evolution
            bp_spec_parents[spec_id] = list(spec_parents)

        #### Offspring Size Calculation ####
        # Determine the amount of offspring for each species as well as the amount of reinitialized blueprints, in case
        # this option is activated. Each species is assigned offspring according to its species share of the total
        # fitness, though minimum offspring constraints are considered. Preprocess by determining the sum of all
        # average fitness and removing the extinct species from the species order
        total_avg_fitness = 0
        for fitness_history in self.pop.bp_species_fitness_history.values():
            total_avg_fitness += fitness_history[self.pop.generation_counter]
        for spec_id in bp_spec_extinct:
            bp_species_ordered.remove(spec_id)

        # Determine the amount of offspring to be reinitialized as the fitness share of the total fitness by the extinct
        # species
        bp_spec_offspring = dict()
        available_bp_pop = self.bp_pop_size
        if self.bp_spec_reinit_extinct and extinct_fitness > 0:
            extinct_fitness_share = extinct_fitness / (total_avg_fitness + extinct_fitness)
            reinit_offspring = int(extinct_fitness_share * available_bp_pop)
            bp_spec_offspring['reinit'] = reinit_offspring
            available_bp_pop -= reinit_offspring

        # Work through each species in order (from least to most fit) and determine the intended size as the species
        # fitness share of the total fitness of the remaining species, applied to the remaining population slots.
        # Assign offspring under the consideration of the minimal offspring constraint and then decrease the total
        # fitness and the remaining population slots.
        for spec_id in bp_species_ordered:
            spec_fitness = self.pop.bp_species_fitness_history[spec_id][self.pop.generation_counter]
            spec_fitness_share = spec_fitness / total_avg_fitness
            spec_intended_size = int(round(spec_fitness_share * available_bp_pop))

            if len(self.pop.bp_species[spec_id]) + self.bp_spec_min_offspring > spec_intended_size:
                bp_spec_offspring[spec_id] = self.bp_spec_min_offspring
                available_bp_pop -= len(self.pop.bp_species[spec_id]) + self.bp_spec_min_offspring
            else:
                bp_spec_offspring[spec_id] = spec_intended_size - len(self.pop.bp_species[spec_id])
                available_bp_pop -= spec_intended_size
            total_avg_fitness -= spec_fitness

        # Return
        # bp_spec_offspring {int: int} associating species id with amount of offspring
        # bp_spec_parents {int: [int]} associating species id with list of potential parent ids for species
        return bp_spec_offspring, bp_spec_parents

    def _select_blueprints_gene_overlap_dynamic(self) -> ({Union[int, str]: int}, {int: int}):
        """"""
        # selection process identical for both variants of blueprint speciation
        return self._select_blueprints_gene_overlap_fixed()
