import json

import tensorflow as tf
from absl import logging

from .codeepneat_model import create_model
from .codeepneat_blueprint import CoDeepNEATBlueprint
from .modules.codeepneat_module_base import CoDeepNEATModuleBase
from ..base_genome import BaseGenome


class CoDeepNEATGenome(BaseGenome):
    """"""

    def __init__(self,
                 genome_id,
                 blueprint,
                 bp_assigned_modules,
                 output_layers,
                 input_shape,
                 dtype,
                 origin_generation):
        """"""
        # Register parameters
        self.genome_id = genome_id
        self.input_shape = input_shape
        self.dtype = dtype
        self.origin_generation = origin_generation

        # Register genotype
        self.blueprint = blueprint
        self.bp_assigned_modules = bp_assigned_modules
        self.output_layers = output_layers

        # Initialize internal variables
        self.fitness = None

        # Create optimizer and model
        self.optimizer = self.blueprint.create_optimizer()
        self.model = create_model(self.blueprint,
                                  self.bp_assigned_modules,
                                  self.output_layers,
                                  self.input_shape,
                                  self.dtype)

    def __call__(self, inputs) -> tf.Tensor:
        """"""
        return self.model.predict(inputs)

    def __str__(self) -> str:
        """"""
        return "CoDeepNEAT Genome | ID: {:>6} | Fitness: {:>6} | Blueprint ID: {:>6} | Module Species: {} | " \
               "Optimizer: {:>6} | Origin Gen: {:>4}".format(self.genome_id,
                                                             self.fitness,
                                                             self.blueprint.get_id(),
                                                             self.blueprint.get_species(),
                                                             self.blueprint.optimizer_factory.get_name(),
                                                             self.origin_generation)

    def save_genotype(self, save_dir_path):
        """"""
        # Set save file name as the genome id and indicate that its the genotype that is being saved
        if save_dir_path[-1] != '/':
            save_dir_path += '/'
        save_file_path = save_dir_path + f"genome_{self.genome_id}_genotype.json"

        # Serializethe assignmend of modules to the bp species for json output
        serialized_bp_assigned_mods = dict()
        for spec, assigned_mod in self.bp_assigned_modules.items():
            serialized_bp_assigned_mods[spec] = assigned_mod.serialize()

        # Use the serialized mod to bp assignment to create a serialization of the whole genome
        serialized_genome = {
            'genome_id': self.genome_id,
            'blueprint': self.blueprint.serialize(),
            'bp_assigned_modules': serialized_bp_assigned_mods,
            'output_layers': self.output_layers,
            'input_shape': self.input_shape,
            'dtype': self.dtype,
            'origin_generation': self.origin_generation
        }

        # Actually save the just serialzied genome as a json file
        with open(save_file_path, 'w') as save_file:
            json.dump(serialized_genome, save_file, indent=4)
        print(f"Saved CoDeepNEAT genome (ID: {self.genome_id}) to file: {save_file_path}")

    def save_model(self, save_dir_path):
        """"""
        logging.warning("CoDeepNEATGenome.save_model() NOT YET IMPLEMENTED")

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_genotype(self) -> (CoDeepNEATBlueprint, {int: CoDeepNEATModuleBase}):
        """"""
        return self.blueprint, self.bp_assigned_modules

    def get_model(self) -> tf.keras.Model:
        """"""
        return self.model

    def get_optimizer(self) -> tf.keras.optimizers.Optimizer:
        """"""
        return self.optimizer

    def get_id(self) -> int:
        return self.genome_id

    def get_fitness(self) -> float:
        return self.fitness
