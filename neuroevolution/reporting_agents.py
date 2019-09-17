import os
from absl import logging


class GenomeRenderAgent:
    def __init__(self, view, render_dir_path):
        self.view = view
        self.render_dir_path = os.path.abspath(render_dir_path)
        if self.render_dir_path[-1] != "/":
            self.render_dir_path += "/"

    def __call__(self, population):
        best_genome = population.get_best_genome()
        file_name = "graph_best_genome_gen_" + str(population.get_generation_counter())
        render_file_path = self.render_dir_path + file_name
        best_genome.visualize(view=self.view, render_file_path=render_file_path)


class PopulationBackupAgent:
    def __init__(self, backup_periodicity, backup_dir_path):
        self.backup_periodicity = backup_periodicity
        self.backup_dir_path = os.path.abspath(backup_dir_path)
        if self.backup_dir_path[-1] != "/":
            self.backup_dir_path += "/"

    def __call__(self, population):
        generation = population.get_generation_counter()
        if generation % self.backup_periodicity == 0:
            logging.info("Automatically backing up population in generation {}...".format(generation))
            backup_file_path = self.backup_dir_path + "population_backup_gen_" + str(generation)
            population.save_population(backup_file_path)
