import tensorflow as tf

from tfne.helper_functions import read_option_from_config


class CoDeepNEATConfigProcessing:
    def _process_config(self):
        """
        Process TFNE CoDeepNEAT compatible config file by reading all parameters as per documentation and saving them
        as instance variables.
        """
        # Read and process the population config values for CoDeepNEAT
        self.bp_pop_size = read_option_from_config(self.config, 'POPULATION', 'bp_pop_size')
        self.mod_pop_size = read_option_from_config(self.config, 'POPULATION', 'mod_pop_size')
        self.genomes_per_bp = read_option_from_config(self.config, 'POPULATION', 'genomes_per_bp')

        # Read and process the config values that concern the genome creation for CoDeepNEAT
        self.dtype = read_option_from_config(self.config, 'GENOME', 'dtype')
        self.available_modules = read_option_from_config(self.config, 'GENOME', 'available_modules')
        self.available_optimizers = read_option_from_config(self.config, 'GENOME', 'available_optimizers')
        self.output_layers = read_option_from_config(self.config, 'GENOME', 'output_layers')

        # Adjust output_layers config to include the configured datatype
        for out_layer in self.output_layers:
            out_layer['config']['dtype'] = self.dtype

        # Read and process the config values that concern the module speciation for CoDeepNEAT
        self.mod_spec_type = read_option_from_config(self.config, 'MODULE_SPECIATION', 'mod_spec_type')
        if self.mod_spec_type == 'basic':
            self.mod_spec_mod_elitism = read_option_from_config(self.config,
                                                                'MODULE_SPECIATION',
                                                                'mod_spec_mod_elitism')
            self.mod_spec_min_offspring = read_option_from_config(self.config,
                                                                  'MODULE_SPECIATION',
                                                                  'mod_spec_min_offspring')
            self.mod_spec_reprod_thres = read_option_from_config(self.config,
                                                                 'MODULE_SPECIATION',
                                                                 'mod_spec_reprod_thres')
        elif self.mod_spec_type == 'param-distance-fixed':
            self.mod_spec_distance = read_option_from_config(self.config,
                                                             'MODULE_SPECIATION',
                                                             'mod_spec_distance')
            self.mod_spec_mod_elitism = read_option_from_config(self.config,
                                                                'MODULE_SPECIATION',
                                                                'mod_spec_mod_elitism')
            self.mod_spec_min_offspring = read_option_from_config(self.config,
                                                                  'MODULE_SPECIATION',
                                                                  'mod_spec_min_offspring')
            self.mod_spec_reprod_thres = read_option_from_config(self.config,
                                                                 'MODULE_SPECIATION',
                                                                 'mod_spec_reprod_thres')
            self.mod_spec_max_stagnation = read_option_from_config(self.config,
                                                                   'MODULE_SPECIATION',
                                                                   'mod_spec_max_stagnation')
            self.mod_spec_species_elitism = read_option_from_config(self.config,
                                                                    'MODULE_SPECIATION',
                                                                    'mod_spec_species_elitism')
            self.mod_spec_rebase_repr = read_option_from_config(self.config,
                                                                'MODULE_SPECIATION',
                                                                'mod_spec_rebase_repr')
            self.mod_spec_reinit_extinct = read_option_from_config(self.config,
                                                                   'MODULE_SPECIATION',
                                                                   'mod_spec_reinit_extinct')
        elif self.mod_spec_type == 'param-distance-dynamic':
            self.mod_spec_species_count = read_option_from_config(self.config,
                                                                  'MODULE_SPECIATION',
                                                                  'mod_spec_species_count')
            self.mod_spec_distance = read_option_from_config(self.config,
                                                             'MODULE_SPECIATION',
                                                             'mod_spec_distance')
            self.mod_spec_mod_elitism = read_option_from_config(self.config,
                                                                'MODULE_SPECIATION',
                                                                'mod_spec_mod_elitism')
            self.mod_spec_min_offspring = read_option_from_config(self.config,
                                                                  'MODULE_SPECIATION',
                                                                  'mod_spec_min_offspring')
            self.mod_spec_reprod_thres = read_option_from_config(self.config,
                                                                 'MODULE_SPECIATION',
                                                                 'mod_spec_reprod_thres')
            self.mod_spec_max_stagnation = read_option_from_config(self.config,
                                                                   'MODULE_SPECIATION',
                                                                   'mod_spec_max_stagnation')
            self.mod_spec_species_elitism = read_option_from_config(self.config,
                                                                    'MODULE_SPECIATION',
                                                                    'mod_spec_species_elitism')
            self.mod_spec_rebase_repr = read_option_from_config(self.config,
                                                                'MODULE_SPECIATION',
                                                                'mod_spec_rebase_repr')
            self.mod_spec_reinit_extinct = read_option_from_config(self.config,
                                                                   'MODULE_SPECIATION',
                                                                   'mod_spec_reinit_extinct')
        else:
            raise NotImplementedError(f"Module speciation type '{self.mod_spec_type}' not yet implemented")

        # Read and process the config values that concern the evolution of modules for CoDeepNEAT
        self.mod_max_mutation = read_option_from_config(self.config, 'MODULE_EVOLUTION', 'mod_max_mutation')
        self.mod_mutation_prob = read_option_from_config(self.config, 'MODULE_EVOLUTION', 'mod_mutation_prob')
        self.mod_crossover_prob = read_option_from_config(self.config, 'MODULE_EVOLUTION', 'mod_crossover_prob')

        # Read and process the config values that concern the blueprint speciation for CoDeepNEAT
        self.bp_spec_type = read_option_from_config(self.config, 'BP_SPECIATION', 'bp_spec_type')
        if self.bp_spec_type == 'basic':
            self.bp_spec_bp_elitism = read_option_from_config(self.config, 'BP_SPECIATION', 'bp_spec_bp_elitism')
            self.bp_spec_min_offspring = read_option_from_config(self.config, 'BP_SPECIATION', 'bp_spec_min_offspring')
            self.bp_spec_reprod_thres = read_option_from_config(self.config, 'BP_SPECIATION', 'bp_spec_reprod_thres')
        elif self.bp_spec_type == 'gene-overlap-fixed':
            self.bp_spec_distance = read_option_from_config(self.config, 'BP_SPECIATION', 'bp_spec_distance')
            self.bp_spec_bp_elitism = read_option_from_config(self.config, 'BP_SPECIATION', 'bp_spec_bp_elitism')
            self.bp_spec_min_offspring = read_option_from_config(self.config, 'BP_SPECIATION', 'bp_spec_min_offspring')
            self.bp_spec_reprod_thres = read_option_from_config(self.config, 'BP_SPECIATION', 'bp_spec_reprod_thres')
            self.bp_spec_max_stagnation = read_option_from_config(self.config,
                                                                  'BP_SPECIATION',
                                                                  'bp_spec_max_stagnation')
            self.bp_spec_species_elitism = read_option_from_config(self.config,
                                                                   'BP_SPECIATION',
                                                                   'bp_spec_species_elitism')
            self.bp_spec_rebase_repr = read_option_from_config(self.config, 'BP_SPECIATION', 'bp_spec_rebase_repr')
            self.bp_spec_reinit_extinct = read_option_from_config(self.config,
                                                                  'BP_SPECIATION',
                                                                  'bp_spec_reinit_extinct')
        elif self.bp_spec_type == 'gene-overlap-dynamic':
            self.bp_spec_species_count = read_option_from_config(self.config, 'BP_SPECIATION', 'bp_spec_species_count')
            self.bp_spec_distance = read_option_from_config(self.config, 'BP_SPECIATION', 'bp_spec_distance')
            self.bp_spec_bp_elitism = read_option_from_config(self.config, 'BP_SPECIATION', 'bp_spec_bp_elitism')
            self.bp_spec_min_offspring = read_option_from_config(self.config, 'BP_SPECIATION', 'bp_spec_min_offspring')
            self.bp_spec_reprod_thres = read_option_from_config(self.config, 'BP_SPECIATION', 'bp_spec_reprod_thres')
            self.bp_spec_max_stagnation = read_option_from_config(self.config,
                                                                  'BP_SPECIATION',
                                                                  'bp_spec_max_stagnation')
            self.bp_spec_species_elitism = read_option_from_config(self.config,
                                                                   'BP_SPECIATION',
                                                                   'bp_spec_species_elitism')
            self.bp_spec_rebase_repr = read_option_from_config(self.config, 'BP_SPECIATION', 'bp_spec_rebase_repr')
            self.bp_spec_reinit_extinct = read_option_from_config(self.config,
                                                                  'BP_SPECIATION',
                                                                  'bp_spec_reinit_extinct')
        else:
            raise NotImplementedError(f"Blueprint speciation type '{self.bp_spec_type}' not yet implemented")

        # Read and process the config values that concern the evolution of blueprints for CoDeepNEAT
        self.bp_max_mutation = read_option_from_config(self.config, 'BP_EVOLUTION', 'bp_max_mutation')
        self.bp_mutation_add_conn_prob = read_option_from_config(self.config,
                                                                 'BP_EVOLUTION',
                                                                 'bp_mutation_add_conn_prob')
        self.bp_mutation_add_node_prob = read_option_from_config(self.config,
                                                                 'BP_EVOLUTION',
                                                                 'bp_mutation_add_node_prob')
        self.bp_mutation_rem_conn_prob = read_option_from_config(self.config,
                                                                 'BP_EVOLUTION',
                                                                 'bp_mutation_rem_conn_prob')
        self.bp_mutation_rem_node_prob = read_option_from_config(self.config,
                                                                 'BP_EVOLUTION',
                                                                 'bp_mutation_rem_node_prob')
        self.bp_mutation_node_spec_prob = read_option_from_config(self.config,
                                                                  'BP_EVOLUTION',
                                                                  'bp_mutation_node_spec_prob')
        self.bp_mutation_optimizer_prob = read_option_from_config(self.config,
                                                                  'BP_EVOLUTION',
                                                                  'bp_mutation_optimizer_prob')
        self.bp_crossover_prob = read_option_from_config(self.config, 'BP_EVOLUTION', 'bp_crossover_prob')

        # Read and process the config values that concern the parameter range of the modules for CoDeepNEAT
        self.available_mod_params = dict()
        for available_mod in self.available_modules:
            # Determine a dict of all supplied configuration values as literal evals
            config_section_str = 'MODULE_' + available_mod.upper()
            if not self.config.has_section(config_section_str):
                raise RuntimeError(f"Module '{available_mod}' marked as available in config does not have an "
                                   f"associated config section defining its parameters")
            mod_section_params = dict()
            for mod_param in self.config.options(config_section_str):
                mod_section_params[mod_param] = read_option_from_config(self.config, config_section_str, mod_param)

            # Assign that dict of all available parameters for the module to the instance variable
            self.available_mod_params[available_mod] = mod_section_params

        # Read and process the config values that concern the parameter range of the optimizers for CoDeepNEAT
        self.available_opt_params = dict()
        for available_opt in self.available_optimizers:
            # Determine a dict of all supplied configuration values as literal evals
            config_section_str = 'OPTIMIZER_' + available_opt.upper()
            if not self.config.has_section(config_section_str):
                raise RuntimeError(f"Optimizer '{available_opt}' marked as available in config does not have an "
                                   f"associated config section defining its parameters")
            opt_section_params = dict()
            for opt_param in self.config.options(config_section_str):
                opt_section_params[opt_param] = read_option_from_config(self.config, config_section_str, opt_param)

            # Assign that dict of all available parameters for the optimizers to the instance variable
            self.available_opt_params[available_opt] = opt_section_params

    def _sanity_check_config(self):
        """
        Perform very basic sanity checks of the TFNE CoDeepNEAT config, as also apparent in the documentation
        """
        # Sanity check [POPULATION] section
        assert self.bp_pop_size > 0 and isinstance(self.bp_pop_size, int)
        assert self.mod_pop_size > 0 and isinstance(self.mod_pop_size, int)
        assert self.genomes_per_bp > 0 and isinstance(self.genomes_per_bp, int)

        # Sanity check [GENOME] section
        assert hasattr(tf.dtypes, self.dtype)
        assert len(self.available_modules) > 0 and isinstance(self.available_modules, list)
        for opt in self.available_optimizers:
            assert hasattr(tf.keras.optimizers, opt)
        for layer in self.output_layers:
            assert tf.keras.layers.deserialize(layer) is not None

        # Sanity check [MODULE_SPECIATION] section
        assert self.mod_spec_type == 'basic' or \
               self.mod_spec_type == 'param-distance-fixed' or \
               self.mod_spec_type == 'param-distance-dynamic'
        if self.mod_spec_type == 'basic':
            assert self.mod_spec_mod_elitism > 0 and isinstance(self.mod_spec_mod_elitism, int)
            assert self.mod_spec_min_offspring >= 0 and isinstance(self.mod_spec_min_offspring, int)
            assert 1.0 >= self.mod_spec_reprod_thres >= 0
        elif self.mod_spec_type == 'param-distance-fixed':
            assert 1.0 >= self.mod_spec_distance >= 0
            assert self.mod_spec_mod_elitism >= 0 and isinstance(self.mod_spec_mod_elitism, int)
            assert self.mod_spec_min_offspring >= 0 and isinstance(self.mod_spec_min_offspring, int)
            assert 1.0 >= self.mod_spec_reprod_thres >= 0
            assert self.mod_spec_max_stagnation > 0 and isinstance(self.mod_spec_max_stagnation, int)
            assert self.mod_spec_species_elitism >= 0 and isinstance(self.mod_spec_species_elitism, int)
            assert isinstance(self.mod_spec_rebase_repr, bool)
            assert isinstance(self.mod_spec_reinit_extinct, bool)
        elif self.mod_spec_type == 'param-distance-dynamic':
            assert self.mod_spec_species_count > 0 and isinstance(self.mod_spec_species_count, int)
            assert 1.0 >= self.mod_spec_distance >= 0
            assert self.mod_spec_mod_elitism >= 0 and isinstance(self.mod_spec_mod_elitism, int)
            assert self.mod_spec_min_offspring >= 0 and isinstance(self.mod_spec_min_offspring, int)
            assert 1.0 >= self.mod_spec_reprod_thres >= 0
            assert self.mod_spec_max_stagnation > 0 and isinstance(self.mod_spec_max_stagnation, int)
            assert self.mod_spec_species_elitism >= 0 and isinstance(self.mod_spec_species_elitism, int)
            assert isinstance(self.mod_spec_rebase_repr, bool)
            assert isinstance(self.mod_spec_reinit_extinct, bool)

        # Sanity check [MODULE_EVOLUTION] section
        assert 1.0 >= self.mod_max_mutation >= 0
        assert 1.0 >= self.mod_mutation_prob >= 0
        assert 1.0 >= self.mod_crossover_prob >= 0
        assert round(self.mod_mutation_prob + self.mod_crossover_prob, 4) == 1.0

        # Sanity check [BP_SPECIATION] section
        assert self.bp_spec_type == 'basic' or \
               self.bp_spec_type == 'gene-overlap-fixed' or \
               self.bp_spec_type == 'gene-overlap-dynamic'
        if self.bp_spec_type == 'basic':
            assert self.bp_spec_bp_elitism > 0 and isinstance(self.bp_spec_bp_elitism, int)
            assert self.bp_spec_min_offspring >= 0 and isinstance(self.bp_spec_min_offspring, int)
            assert 1.0 >= self.bp_spec_reprod_thres >= 0
        elif self.bp_spec_type == 'gene-overlap-fixed':
            assert 1.0 >= self.bp_spec_distance >= 0
            assert self.bp_spec_bp_elitism >= 0 and isinstance(self.bp_spec_bp_elitism, int)
            assert self.bp_spec_min_offspring >= 0 and isinstance(self.bp_spec_min_offspring, int)
            assert 1.0 >= self.bp_spec_reprod_thres >= 0
            assert self.bp_spec_max_stagnation > 0 and isinstance(self.bp_spec_max_stagnation, int)
            assert self.bp_spec_species_elitism >= 0 and isinstance(self.bp_spec_species_elitism, int)
            assert isinstance(self.bp_spec_rebase_repr, bool)
            assert isinstance(self.bp_spec_reinit_extinct, bool)
        elif self.bp_spec_type == 'gene-overlap-dynamic':
            assert self.bp_spec_species_count > 0 and isinstance(self.bp_spec_species_count, int)
            assert 1.0 >= self.bp_spec_distance >= 0
            assert self.bp_spec_bp_elitism >= 0 and isinstance(self.bp_spec_bp_elitism, int)
            assert self.bp_spec_min_offspring >= 0 and isinstance(self.bp_spec_min_offspring, int)
            assert 1.0 >= self.bp_spec_reprod_thres >= 0
            assert self.bp_spec_max_stagnation > 0 and isinstance(self.bp_spec_max_stagnation, int)
            assert self.bp_spec_species_elitism >= 0 and isinstance(self.bp_spec_species_elitism, int)
            assert isinstance(self.bp_spec_rebase_repr, bool)
            assert isinstance(self.bp_spec_reinit_extinct, bool)

        # Sanity check [BP_EVOLUTION] section
        assert 1.0 >= self.bp_max_mutation >= 0
        assert 1.0 >= self.bp_mutation_add_conn_prob >= 0
        assert 1.0 >= self.bp_mutation_add_node_prob >= 0
        assert 1.0 >= self.bp_mutation_rem_conn_prob >= 0
        assert 1.0 >= self.bp_mutation_rem_node_prob >= 0
        assert 1.0 >= self.bp_mutation_node_spec_prob > 0
        assert 1.0 >= self.bp_mutation_optimizer_prob >= 0
        assert 1.0 >= self.bp_crossover_prob >= 0
        assert round(self.bp_mutation_add_conn_prob + self.bp_mutation_add_node_prob + self.bp_mutation_rem_conn_prob
                     + self.bp_mutation_rem_node_prob + self.bp_mutation_node_spec_prob + self.bp_crossover_prob
                     + self.bp_mutation_optimizer_prob, 4) == 1.0
