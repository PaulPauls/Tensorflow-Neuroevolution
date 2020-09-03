import statistics

from ..base_population import BasePopulation


class CoDeepNEATPopulation(BasePopulation):
    """
    Population class of the CoDeepNEAT algorithm that holds all relevant population information in a single place to
    ease summary, serialization and deserialization.
    """

    def __init__(self, initial_state=None):
        """
        Initializes all variables of a CoDeepNEAT population either to None/default values or to an initial state if
        such is supplied (usually when deserializing population)
        @param initial_state: dict object holding keys and values to all population variables
        """
        # Declare internal variables of the CoDeepNEAT population
        self.generation_counter = None
        self.best_genome = None
        self.best_fitness = None

        # Declare and initialize internal variables concerning the module population of the CoDeepNEAT algorithm
        self.modules = dict()
        self.mod_species = dict()
        self.mod_species_repr = dict()
        self.mod_species_fitness_history = dict()
        self.mod_species_counter = 0

        # Declare and initialize internal variables concerning the blueprint population of the CoDeepNEAT algorithm
        self.blueprints = dict()
        self.bp_species = dict()
        self.bp_species_repr = dict()
        self.bp_species_fitness_history = dict()
        self.bp_species_counter = 0

        # If an initial state is supplied, then the population was deserialized. Recreate this initial state.
        if initial_state is not None:
            self.generation_counter = initial_state['generation_counter']
            self.best_genome = initial_state['best_genome']
            self.best_fitness = initial_state['best_fitness']
            self.modules = initial_state['modules']
            self.mod_species = initial_state['mod_species']
            self.mod_species_repr = initial_state['mod_species_repr']
            self.mod_species_fitness_history = initial_state['mod_species_fitness_history']
            self.mod_species_counter = initial_state['mod_species_counter']
            self.blueprints = initial_state['blueprints']
            self.bp_species = initial_state['bp_species']
            self.bp_species_repr = initial_state['bp_species_repr']
            self.bp_species_fitness_history = initial_state['bp_species_fitness_history']
            self.bp_species_counter = initial_state['bp_species_counter']

    def summarize_population(self):
        """
        Prints the current state of all CoDeepNEAT population variables to stdout in a formatted and clear manner
        """
        # Determine average fitness of all blueprints
        bp_fitness_list = [self.blueprints[bp_id].get_fitness() for bp_id in self.blueprints]
        blueprints_avg_fitness = round(statistics.mean(bp_fitness_list), 4)

        # Determine best id of each blueprint species
        bp_species_best_id = dict()
        for spec_id, spec_bp_ids in self.bp_species.items():
            spec_bp_ids_sorted = sorted(spec_bp_ids, key=lambda x: self.blueprints[x].get_fitness(), reverse=True)
            bp_species_best_id[spec_id] = spec_bp_ids_sorted[0]

        # Determine average fitness of all modules
        mod_fitness_list = [self.modules[mod_id].get_fitness() for mod_id in self.modules]
        modules_avg_fitness = round(statistics.mean(mod_fitness_list), 4)

        # Determine best id of each module species
        mod_species_best_id = dict()
        for spec_id, spec_mod_ids in self.mod_species.items():
            spec_mod_ids_sorted = sorted(spec_mod_ids, key=lambda x: self.modules[x].get_fitness(), reverse=True)
            mod_species_best_id[spec_id] = spec_mod_ids_sorted[0]

        # Print summary header
        print("\n\n\n\033[1m{}  Population Summary  {}\n\n"
              "Generation: {:>4}  ||  Best Genome Fitness: {:>8}  ||  Avg Blueprint Fitness: {:>8}  ||  "
              "Avg Module Fitness: {:>8}\033[0m\n"
              "Best Genome: {}\n"
              .format('#' * 60,
                      '#' * 60,
                      self.generation_counter,
                      self.best_fitness,
                      blueprints_avg_fitness,
                      modules_avg_fitness,
                      self.best_genome))

        # Print summary of blueprint species
        print("\033[1mBlueprint Species       || Blueprint Species Avg Fitness       || Blueprint Species Size\033[0m")
        for spec_id, spec_fitness_hisotry in self.bp_species_fitness_history.items():
            print("{:>6}                  || {:>8}                            || {:>8}"
                  .format(spec_id,
                          spec_fitness_hisotry[self.generation_counter],
                          len(self.bp_species[spec_id])))
            print(f"Best BP of Species {spec_id}    || {self.blueprints[bp_species_best_id[spec_id]]}")

        # Print summary of module species
        print("\n\033[1mModule Species          || Module Species Avg Fitness          || Module Species Size\033[0m")
        for spec_id, spec_fitness_hisotry in self.mod_species_fitness_history.items():
            print("{:>6}                  || {:>8}                            || {:>8}"
                  .format(spec_id,
                          spec_fitness_hisotry[self.generation_counter],
                          len(self.mod_species[spec_id])))
            print(f"Best Mod of Species {spec_id}   || {self.modules[mod_species_best_id[spec_id]]}")

        # Print summary footer
        print("\n\033[1m" + '#' * 142 + "\033[0m\n")

    def serialize(self) -> dict:
        """
        Serializes all CoDeepNEAT population variables to a json compatible dictionary and returns it
        @return: serialized population variables as a json compatible dict
        """
        # Serialize all modules
        serialized_modules = dict()
        for mod_id, module in self.modules.items():
            serialized_modules[mod_id] = module.serialize()

        # Serialize all blueprints
        serialized_blueprints = dict()
        for bp_id, blueprint in self.blueprints.items():
            serialized_blueprints[bp_id] = blueprint.serialize()

        # Use serialized module and blueprint population and extend it by population internal evolution information
        serialized_population = {
            'population_type': 'CoDeepNEAT',
            'generation_counter': self.generation_counter,
            'modules': serialized_modules,
            'mod_species': self.mod_species,
            'mod_species_repr': self.mod_species_repr if self.mod_species_repr else None,
            'mod_species_fitness_history': self.mod_species_fitness_history,
            'mod_species_counter': self.mod_species_counter,
            'blueprints': serialized_blueprints,
            'bp_species': self.bp_species,
            'bp_species_repr': self.bp_species_repr if self.bp_species_repr else None,
            'bp_species_fitness_history': self.bp_species_fitness_history,
            'bp_species_counter': self.bp_species_counter,
            'best_genome': self.best_genome.serialize(),
            'best_fitness': self.best_fitness
        }

        return serialized_population
