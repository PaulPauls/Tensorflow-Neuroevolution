import statistics
import logging


class CoDeepNEATSpeciationMOD:
    def _speciate_modules_basic(self, mod_spec_parents, new_module_ids):
        """"""
        ### Removal of Parental But Not Elite Modules ###
        # Remove modules from module container that served as parents and were kept, though do not belong to any species
        for spec_id, spec_parents in mod_spec_parents.items():
            spec_elites = self.pop.mod_species[spec_id]
            for mod_id in spec_parents:
                if mod_id not in spec_elites:
                    del self.pop.modules[mod_id]

        ### Species Assignment ###
        # Basic speciation assigns each new module to the species with the according module type, as for each module
        # type there is only one species. Preprocess species by creating the type to id association
        species_type_to_id = dict()
        for spec_id, spec_mod_ids in self.pop.mod_species.items():
            species_type = self.pop.modules[spec_mod_ids[0]].get_module_type()
            species_type_to_id[species_type] = spec_id

        for mod_id in new_module_ids:
            module_type = self.pop.modules[mod_id].get_module_type()
            according_mod_spec_id = species_type_to_id[module_type]
            self.pop.mod_species[according_mod_spec_id].append(mod_id)

    def _speciate_modules_param_distance_fixed(self, mod_spec_parents, new_module_ids):
        """"""
        ### Removal of Parental But Not Elite Modules ###
        # Remove modules from module container that served as parents and were kept, though do not belong to any species
        for spec_id, spec_parents in mod_spec_parents.items():
            spec_elites = self.pop.mod_species[spec_id]
            for mod_id in spec_parents:
                if mod_id not in spec_elites:
                    del self.pop.modules[mod_id]

        ### Species Assignment ###
        # Traverse all new module ids, determine their type and compare their parameter distance with other species of
        # that type. If the distance to one species of the same type is below the config specified 'mod_spec_distance'
        # then assign the new module to that species. If not, create a new species. Create a preprocessed dict that
        # lists all species of one type as only species with the same type are relevant for comparison.
        species_type_to_id = dict()
        for spec_id, spec_mod_repr_id in self.pop.mod_species_repr.items():
            species_type = self.pop.modules[spec_mod_repr_id].get_module_type()
            if species_type in species_type_to_id:
                species_type_to_id[species_type].append(spec_id)
            else:
                species_type_to_id[species_type] = [spec_id]

        min_spec_size = self.mod_spec_mod_elitism + self.mod_spec_min_offspring + 1
        for mod_id in new_module_ids:
            module_type = self.pop.modules[mod_id].get_module_type()

            # Calculate the distance of the module to each species representative and associate each species with its
            # distance in the module_spec_distances dict
            module_spec_distances = dict()
            for spec_mod_type, spec_ids in species_type_to_id.items():
                if module_type != spec_mod_type:
                    continue

                for spec_id in spec_ids:
                    spec_mod_repr = self.pop.modules[self.pop.mod_species_repr[spec_id]]
                    module_spec_distances[spec_id] = spec_mod_repr.get_distance(self.pop.modules[mod_id])

            # Determine species whose representative has the minimum distance to the new module. If that minimum
            # distance is lower than the config set module species distance, assign the new module to that species.
            # If the minimum distance is higher than the module species distance, create a new species with the new
            # module as the representative, assuming the population size allows for it.
            min_distance_spec = min(module_spec_distances, key=module_spec_distances.get)
            if module_spec_distances[min_distance_spec] <= self.mod_spec_distance:
                self.pop.mod_species[min_distance_spec].append(mod_id)
            elif module_spec_distances[min_distance_spec] > self.mod_spec_distance \
                    and min_spec_size * len(self.pop.mod_species) >= self.mod_pop_size:
                logging.warning(f"Warning: New Module (#{mod_id}) has sufficient distance to other species"
                                f"representatives but has been assigned to species {min_distance_spec} as the"
                                f"population size does not allow for more species.")
                self.pop.mod_species[min_distance_spec].append(mod_id)
            else:
                # Create a new species with the new module as the representative
                self.pop.mod_species_counter += 1
                self.pop.mod_species[self.pop.mod_species_counter] = [mod_id]
                self.pop.mod_species_repr[self.pop.mod_species_counter] = mod_id
                species_type_to_id[module_type].append(self.pop.mod_species_counter)

    def _speciate_modules_param_distance_dynamic(self, mod_spec_parents, new_module_ids):
        """"""
        # Perform param-distance-fixed speciation as identical to dynamic variant and subsequently adjust distance
        self._speciate_modules_param_distance_fixed(mod_spec_parents, new_module_ids)

        ### Dynamic Adjustment of Species Distance ###
        # If the species count is too low, decrease the species distance by 5 percent. If the species count is too
        # high, determine the distances of each species representative to all other species representatives and choose
        # the distance that would set the species count right. Average that optimal distance for each species repr out
        # to get the new species distance.
        if len(self.pop.mod_species) < self.mod_spec_species_count:
            self.mod_spec_distance = self.mod_spec_distance * 0.95
        elif len(self.pop.mod_species) > self.mod_spec_species_count:
            optimal_spec_distance_per_species = list()
            for spec_id, spec_mod_repr_id in self.pop.mod_species_repr.items():
                mod_repr = self.pop.modules[spec_mod_repr_id]
                # Determine distance of species repr to all other species repr
                other_spec_mod_repr_ids = [mod_id for mod_id in self.pop.mod_species_repr.values()
                                           if mod_id != spec_mod_repr_id]
                sorted_distances_to_other_specs = sorted([mod_repr.get_distance(self.pop.modules[other_mod_id])
                                                          for other_mod_id in other_spec_mod_repr_ids])
                # Set optimal distance of current species repr such that it conforms to 'mod_spec_species_count' by
                # choosing the distance that would result in only the desired species count for the current
                # representative
                optimal_spec_distance = sorted_distances_to_other_specs[self.mod_spec_species_count - 1]
                optimal_spec_distance_per_species.append(optimal_spec_distance)

            # Average out all optimal distances for each species repr to get the new distance
            self.mod_spec_distance = statistics.mean(optimal_spec_distance_per_species)
