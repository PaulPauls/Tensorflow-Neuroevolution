import os
from absl import logging


class GenomeRenderAgent:
    """
    Reporting Agent that visualizes the best performing genome of the current generation in a specified periodicity and
    to the specified directory, with an optional parameter to view the rendering after creation. The Genome Render agent
    utilizes the genome's 'visualize' function to render it. The Reporting Agent is automatically called by the
    evolution engine.
    """

    def __init__(self, periodicity, view, render_dir_path):
        """
        :param periodicity: int; Interval in between generations at which the GenomeRenderAgent should be executed
        :param view: bool; Flag if genome visualization should be displayed after creation
        :param render_dir_path: string of directory path, specifying where the genome render should be saved
        """
        self.periodicity = periodicity
        self.view = view
        self.render_dir_path = os.path.abspath(render_dir_path)
        if self.render_dir_path[-1] != "/":
            self.render_dir_path += "/"
        if not os.path.exists(self.render_dir_path):
            os.mkdir(self.render_dir_path)

    def __call__(self, population):
        generation = population.get_generation_counter()
        if generation % self.periodicity == 0:
            best_genome = population.get_best_genome()
            filename = "graph_best_genome_gen_" + str(generation)
            logging.info("Automatically rendering best genome from generation {} to {}..."
                         .format(generation, self.render_dir_path + filename))
            best_genome.visualize(view=self.view, filename=filename, render_dir_path=self.render_dir_path)

    def log_parameters(self):
        logging.debug("Genome Render Agent parameter: periodicity = {}".format(self.periodicity))
        logging.debug("Genome Render Agent parameter: view = {}".format(self.view))
        logging.debug("Genome Render Agent parameter: render_dir_path = {}".format(self.render_dir_path))


class PopulationBackupAgent:
    """
    Reporting Agent that backs-up the population in a specified periodicity and to the specified directory by using
    the population's 'save_population()' function, which in turn serializes the whole population and the evolutionary
    state and saves it as a json file. The Reporting Agent is automatically called by the evolution engine.
    """

    def __init__(self, periodicity, backup_dir_path):
        """
        :param periodicity: int; Interval in between generations at which the GenomeRenderAgent should be executed
        :param backup_dir_path: tring of directory path, specifying where the serialized population should be saved
        """
        self.periodicity = periodicity
        self.backup_dir_path = os.path.abspath(backup_dir_path)
        if self.backup_dir_path[-1] != "/":
            self.backup_dir_path += "/"
        if not os.path.exists(self.backup_dir_path):
            os.mkdir(self.backup_dir_path)

    def __call__(self, population):
        generation = population.get_generation_counter()
        if generation % self.periodicity == 0:
            backup_file_path = "{}population_backup_gen_{}.json".format(self.backup_dir_path, generation)
            logging.info("Automatically backing up population from generation {} to {}..."
                         .format(generation, backup_file_path))
            population.save_population(backup_file_path)

    def log_parameters(self):
        logging.debug("Population Backup Agent parameter: periodicity = {}".format(self.periodicity))
        logging.debug("Population Backup Agent parameter: backup_dir_path = {}".format(self.backup_dir_path))
