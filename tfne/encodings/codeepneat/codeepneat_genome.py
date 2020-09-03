import os
import json
import tempfile
import platform
import subprocess

import tensorflow as tf

from .codeepneat_model import CoDeepNEATModel
from .codeepneat_blueprint import CoDeepNEATBlueprint
from .modules import CoDeepNEATModuleBase
from tfne.encodings.base_genome import BaseGenome


class CoDeepNEATGenome(BaseGenome,
                       CoDeepNEATModel):
    """
    CoDeepNEAT genome that encapsulates the genotype and its associated phenotype, being the translated TF model. It
    furthermore encapsulates the phenotype visualization as well as serialization and genotype saving functionality.
    """

    def __init__(self,
                 genome_id,
                 blueprint,
                 bp_assigned_modules,
                 output_layers,
                 input_shape,
                 dtype,
                 origin_generation):
        """
        Create CoDeepNEAT genome by saving the associated genotype parameters as well as additional information like
        dtype and origin generation. Then create TF model from genotype.
        @param genome_id: int of unique genome ID
        @param blueprint: CoDeepNEAT blueprint instance
        @param bp_assigned_modules: dict associating each BP species with a CoDeepNEAT module instance
        @param output_layers: string of TF deserializable layers serving as output
        @param input_shape: int-tuple specifying the input shape the genome model has to adhere to
        @param dtype: string of TF dtype
        @param origin_generation: int, specifying the evolution generation at which the genome was created
        """
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
        self.model = None
        self.optimizer = self.blueprint.create_optimizer()
        self._create_model()

    def __call__(self, inputs) -> tf.Tensor:
        """"""
        return self.model(inputs)

    def __str__(self) -> str:
        """
        @return: string representation of the genome
        """
        return "CoDeepNEAT Genome | ID: {:>6} | Fitness: {:>6} | Blueprint ID: {:>6} | Module Species: {} | " \
               "Optimizer: {:>6} | Origin Gen: {:>4}".format(self.genome_id,
                                                             'None' if self.fitness is None else self.fitness,
                                                             self.blueprint.get_id(),
                                                             self.blueprint.get_species(),
                                                             self.blueprint.optimizer_factory.get_name(),
                                                             self.origin_generation)

    def visualize(self, show=True, save_dir_path=None, **kwargs) -> str:
        """
        Visualize the CoDeepNEAT genome through dot. If 'show' flag is set to true, display the genome after rendering.
        If 'save_dir_path' is supplied, save the rendered genome as file to that directory.
        Return the saved file path as string.
        @param show: bool flag, indicating whether the rendered genome should be displayed or not
        @param save_dir_path: string of the save directory path the rendered genome should be saved to.
        @param kwargs: Optional additional arguments for tf.keras.utils.plot_model()
        @return: string of the file path to which the rendered genome has been saved to
        """
        # Check if save_dir_path is supplied and if it is supplied in the correct format. If not correct format or None
        # supplied create a new save_dir_path. Ensure that the save_dir_path exists by creating the directories.
        if save_dir_path is None:
            save_dir_path = tempfile.gettempdir()
        if save_dir_path[-1] != '/':
            save_dir_path += '/'
        os.makedirs(save_dir_path, exist_ok=True)

        # Set save file name as the genome id and indicate that its the model being plotted
        save_file_path = save_dir_path + f"genome_{self.genome_id}_model.svg"

        # Adjust plotting model parameters to show shapes but not layers names by default if none of those parameters
        # are supplied. Set DPI to None regardless of supplied parameters as keras utils API is bugged when supplying
        # DPI for svg plots. See Tensorflow issue: https://github.com/tensorflow/tensorflow/issues/42150
        if 'show_shapes' not in kwargs:
            kwargs['show_shapes'] = True
        if 'show_layer_names' not in kwargs:
            kwargs['show_layer_names'] = False
        kwargs['dpi'] = None

        # Create plot of model through keras util
        tf.keras.utils.plot_model(model=self.model, to_file=save_file_path, **kwargs)

        # If visualization is set to show, open it in the default image program
        if show and platform.system() == 'Windows':
            save_file_normpath = os.path.normpath(save_file_path)
            os.startfile(save_file_normpath)
        elif show:
            subprocess.Popen(['xdg-open', save_file_path])

        # Return the file path to which the genome plot was saved
        return save_file_path

    def serialize(self) -> dict:
        """
        @return: serialized constructor variables of the genome as json compatible dict
        """
        # Serialize the assignment of modules to the bp species for json output
        serialized_bp_assigned_mods = dict()
        for spec, assigned_mod in self.bp_assigned_modules.items():
            serialized_bp_assigned_mods[spec] = assigned_mod.serialize()

        # Use the serialized mod to bp assignment to create a serialization of the whole genome
        serialized_genome = {
            'genome_type': 'CoDeepNEAT',
            'genome_id': self.genome_id,
            'fitness': self.fitness,
            'blueprint': self.blueprint.serialize(),
            'bp_assigned_modules': serialized_bp_assigned_mods,
            'output_layers': self.output_layers,
            'input_shape': self.input_shape,
            'dtype': self.dtype,
            'origin_generation': self.origin_generation
        }

        return serialized_genome

    def save_genotype(self, save_dir_path) -> str:
        """
        Save genotype of CoDeepNEAT genome to 'save_dir_path' directory. Return file path to which the genotype has
        been saved to as string.
        @param save_dir_path: string of the save directory path the genotype should be saved to
        @return: string of the file path to which the genotype has been saved to
        """
        # Set save file name as the genome id and indicate that its the genotype that is being saved. Ensure that the
        # save_dir_path exists by creating the directories.
        if save_dir_path[-1] != '/':
            save_dir_path += '/'
        os.makedirs(save_dir_path, exist_ok=True)
        save_file_path = save_dir_path + f"genome_{self.genome_id}_genotype.json"

        # Create serialization of the genome
        serialized_genome = self.serialize()

        # Actually save the just serialzied genome as a json file
        with open(save_file_path, 'w') as save_file:
            json.dump(serialized_genome, save_file, indent=4)
        print(f"Saved CoDeepNEAT genome (ID: {self.genome_id}) to file: {save_file_path}")

        # Return the file path to which the genome was saved
        return save_file_path

    def save_model(self, file_path, **kwargs):
        """"""
        self.model.save(filepath=file_path, **kwargs)

    def set_fitness(self, fitness):
        self.fitness = fitness

    def get_genotype(self) -> (CoDeepNEATBlueprint, {int: CoDeepNEATModuleBase}, [dict]):
        """"""
        return self.blueprint, self.bp_assigned_modules, self.output_layers

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
