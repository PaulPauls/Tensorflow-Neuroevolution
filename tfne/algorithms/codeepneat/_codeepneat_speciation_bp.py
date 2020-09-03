import warnings
import statistics


class CoDeepNEATSpeciationBP:
    def _speciate_blueprints_basic(self, bp_spec_parents, new_blueprint_ids):
        """"""
        ### Removal of Parental But Not Elite Blueprints ###
        # Remove blueprints from blueprint container that served as parents and were kept, though do not belong to any
        # species
        for spec_id, spec_parents in bp_spec_parents.items():
            spec_elites = self.pop.bp_species[spec_id]
            for bp_id in spec_parents:
                if bp_id not in spec_elites:
                    del self.pop.blueprints[bp_id]

        ### Species Assignment ###
        # Basic speciation assigns each new blueprint to species 1, as the only existing species
        self.pop.bp_species[1] += new_blueprint_ids

    def _speciate_blueprints_gene_overlap_fixed(self, bp_spec_parents, new_blueprint_ids):
        """"""
        ### Removal of Parental But Not Elite Blueprints ###
        # Remove blueprints from blueprint container that served as parents and were kept, though do not belong to any
        # species
        for spec_id, spec_parents in bp_spec_parents.items():
            spec_elites = self.pop.bp_species[spec_id]
            for bp_id in spec_parents:
                if bp_id not in spec_elites:
                    del self.pop.blueprints[bp_id]

        ### Species Assignment ###
        # Traverse all new blueprint ids and compare the gene overlap distance with all species representatives. If the
        # distance is below the config specified 'bp_spec_distance' then assign the new blueprint to that species. If
        # not, create a new species.
        min_spec_size = self.bp_spec_bp_elitism + self.bp_spec_min_offspring + 1
        for bp_id in new_blueprint_ids:
            # Calculate the distance of the blueprint to each species representative and associate each species with its
            # distance in the bp_spec_distances dict
            bp_spec_distances = dict()
            for spec_id, spec_bp_repr_id in self.pop.bp_species_repr.items():
                spec_bp_repr = self.pop.blueprints[spec_bp_repr_id]
                bp_spec_distances[spec_id] = spec_bp_repr.calculate_gene_distance(self.pop.blueprints[bp_id])

            # Determine species whose representative has the minimum distance to the new blueprint. If that minimum
            # distance is lower than the config set blueprint species distance, assign the new blueprint to that
            # species. If the minimum distance is higher than the blueprint species distance, create a new species with
            # the new blueprint as the representative, assuming the population size allows for it.
            min_distance_spec = min(bp_spec_distances, key=bp_spec_distances.get)
            if bp_spec_distances[min_distance_spec] <= self.bp_spec_distance:
                self.pop.bp_species[min_distance_spec].append(bp_id)
            elif bp_spec_distances[min_distance_spec] > self.bp_spec_distance \
                    and min_spec_size * len(self.pop.bp_species) >= self.bp_pop_size:
                warnings.warn(f"Warning: New Blueprint (#{bp_id}) has sufficient distance to other species"
                              f"representatives but has been assigned to species {min_distance_spec} as the"
                              f"population size does not allow for more species.", UserWarning)
                self.pop.bp_species[min_distance_spec].append(bp_id)
            else:
                # Create a new species with the new blueprint as the representative
                self.pop.bp_species_counter += 1
                self.pop.bp_species[self.pop.bp_species_counter] = [bp_id]
                self.pop.bp_species_repr[self.pop.bp_species_counter] = bp_id

    def _speciate_blueprints_gene_overlap_dynamic(self, bp_spec_parents, new_blueprint_ids):
        """"""
        # Perform gene-overlap-dynamic speciation as identical to dynamic variant and subsequently adjust distance
        self._speciate_blueprints_gene_overlap_fixed(bp_spec_parents, new_blueprint_ids)

        ### Dynamic Adjustment of Species Distance ###
        # If the species count is too low, decrease the species distance by 5 percent. If the species count is too
        # high, determine the distances of each species representative to all other species representatives and choose
        # the distance that would set the species count right. Average that optimal distance for each species repr out
        # to get the new species distance.
        if len(self.pop.bp_species) < self.bp_spec_species_count:
            self.bp_spec_distance = self.bp_spec_distance * 0.95
        elif len(self.pop.bp_species) > self.bp_spec_species_count:
            optimal_spec_distance_per_species = list()
            for spec_id, spec_bp_repr_id in self.pop.bp_species_repr.items():
                bp_repr = self.pop.blueprints[spec_bp_repr_id]
                # Determine distance of species repr to all other species repr
                other_spec_bp_repr_ids = [bp_id for bp_id in self.pop.bp_species_repr.values()
                                          if bp_id != spec_bp_repr_id]
                distances_to_other_specs = [bp_repr.calculate_gene_distance(self.pop.blueprints[other_bp_id])
                                            for other_bp_id in other_spec_bp_repr_ids]
                sorted_distances_to_other_specs = sorted(distances_to_other_specs)
                # Set optimal distance of current species repr such that it conforms to 'bp_spec_species_count' by
                # choosing the distance that would result in only the desired species count for the current
                # representative
                optimal_spec_distance = sorted_distances_to_other_specs[self.bp_spec_species_count - 1]
                optimal_spec_distance_per_species.append(optimal_spec_distance)

            # Average out all optimal distances for each species repr to get the new distance
            self.bp_spec_distance = statistics.mean(optimal_spec_distance_per_species)
