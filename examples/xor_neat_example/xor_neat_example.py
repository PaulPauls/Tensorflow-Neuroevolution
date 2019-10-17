import tensorflow as tf
from absl import logging

import neuroevolution as ne


def main():
    """
    Basic example of the Tensorflow-Neuroevolution framework setting up all required elements of the neuroevolutionary
    process to evolve a genome to solve the XOR environment by evolving it with the NEAT algorithm.
    """

    logging.set_verbosity(logging.DEBUG)
    logging.info("Using TF Version {}".format(tf.__version__))
    assert tf.__version__[0] == '2'  # Assert that TF 2.x is used

    config = ne.load_config('./xor_neat_example.cfg')

    environment = ne.environments.XOREnvironment()
    ne_algorithm = ne.algorithms.NEAT(config, dtype=tf.float32, run_eagerly=False)

    population = ne.Population(config, ne_algorithm)
    engine = ne.EvolutionEngine(population, environment)

    genome_render_agent = ne.GenomeRenderAgent(periodicity=5, view=False, render_dir_path="./genome_renders")
    population_backup_agent = ne.PopulationBackupAgent(periodicity=50, backup_dir_path="./population_backups")
    reporting_agents = (genome_render_agent, population_backup_agent)
    best_genome = engine.train(reporting_agents=reporting_agents)

    if best_genome is not None:
        environment.replay_genome(best_genome)
        logging.info("Best Genome returned by evolution:\n{}".format(best_genome))
        logging.info("Visualizing best genome returned by evolution...")
        # best_genome.visualize()
        logging.warning("DirectEncodingGenome.visualize() not yet implemented")
    else:
        logging.info("Evolution of population did not return a valid genome")


if __name__ == '__main__':
    main()
