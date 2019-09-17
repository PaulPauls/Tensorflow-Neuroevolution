from absl import logging


class GenomeRenderAgent:
    def __init__(self, render_dir_path):
        raise NotImplementedError()

    def __call__(self, population):
        raise NotImplementedError()


class PopulationBackupAgent:
    def __init__(self, backup_periodicity, backup_dir_path):
        self.backup_periodicity = backup_periodicity
        self.backup_dir_path = backup_dir_path
        if self.backup_dir_path[-1] != "/":
            self.backup_dir_path += "/"

    def __call__(self, population):
        generation = population.get_generation_counter()
        if generation % self.backup_periodicity == 0:
            logging.info("Automatically backing up population in generation {}...".format(generation))
            save_path = self.backup_dir_path + "population_backup_gen_" + str(generation)
            population.save_population(save_path)
