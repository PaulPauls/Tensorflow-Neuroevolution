from absl import app, flags, logging

import tfne

flags.DEFINE_string('logging_level',
                    default=None, help='TODO')
flags.DEFINE_string('config_file',
                    default=None, help='TODO')
flags.DEFINE_string('backup_dir',
                    default=None, help='TODO')
flags.DEFINE_integer('num_cpus',
                     default=None, help='TODO')
flags.DEFINE_integer('num_gpus',
                     default=None, help='TODO')
flags.DEFINE_integer('max_generations',
                     default=None, help='TODO')
flags.DEFINE_float('max_fitness',
                   default=None, help='TODO')


def codeepneat_xor_example(_):
    """"""
    # Set standard configuration specific to TFNE but not the neuroevolution process
    logging_level = logging.INFO
    config_file_path = './codeepneat_xor_example_config.cfg'
    backup_dir_path = './population_backups/'
    num_cpus = None
    num_gpus = None
    max_generations = 100
    max_fitness = None

    # Read in optionally supplied flags, changing the just set standard configuration
    if flags.FLAGS.logging_level is not None:
        logging_level = flags.FLAGS.logging_level
    if flags.FLAGS.config_file is not None:
        config_file_path = flags.FLAGS.config_file
    if flags.FLAGS.backup_dir is not None:
        backup_dir_path = flags.FLAGS.backup_dir
    if flags.FLAGS.num_cpus is not None:
        num_cpus = flags.FLAGS.num_cpus
    if flags.FLAGS.num_gpus is not None:
        num_gpus = flags.FLAGS.num_gpus
    if flags.FLAGS.max_generations is not None:
        max_generations = flags.FLAGS.max_generations
    if flags.FLAGS.max_fitness is not None:
        max_fitness = flags.FLAGS.max_fitness

    # Set logging, parse config
    logging.set_verbosity(logging_level)
    config = tfne.parse_configuration(config_file_path)

    # Set (not initialize) the environment and initialize the specific NE algorithm
    environment = tfne.environments.XOREnvironment
    ne_algorithm = tfne.CoDeepNEAT(config, environment)

    # Initialize evolution engine and supply config as well as initialized NE elements
    engine = tfne.EvolutionEngine(ne_algorithm=ne_algorithm,
                                  backup_dir_path=backup_dir_path,
                                  num_cpus=num_cpus,
                                  num_gpus=num_gpus,
                                  max_generations=max_generations,
                                  max_fitness=max_fitness)

    # Start training process, returning the best genome when training ends
    best_genome = engine.train()

    # Show string representation of best genome, visualize it and then save it
    print("Best Genome returned by evolution:\n")
    print(best_genome)
    best_genome.save_genotype(save_dir_path='./')
    best_genome.save_model(save_dir_path='./')


if __name__ == '__main__':
    app.run(codeepneat_xor_example)
