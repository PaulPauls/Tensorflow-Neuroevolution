import os
import tempfile

# Deactivate GPUs as pytest seems very error-prone in combination with Tensorflow
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

import tfne


def sanity_check_algorithm_state(ne_algorithm):
    """
    Very basic sanity check as the purpose of the pytest checks is the run of the evolutionary loops. If there are some
    bugs in the evolutionary process the complex logic will fail. Therefore there is not much purpose in doing extensive
    asserts after the evolutionary process succeded.
    """
    best_genome = ne_algorithm.get_best_genome()
    assert 100 >= best_genome.get_fitness() > 0


def test_codeepneat_1():
    # Create test config
    config = tfne.parse_configuration(os.path.dirname(__file__) + '/test_codeepneat_1_config.cfg')
    environment = tfne.environments.XOREnvironment(weight_training=True, config=config)
    ne_algorithm = tfne.algorithms.CoDeepNEAT(config)

    # Start test
    engine = tfne.EvolutionEngine(ne_algorithm=ne_algorithm,
                                  environment=environment,
                                  backup_dir_path=tempfile.gettempdir(),
                                  max_generations=10,
                                  max_fitness=None)
    engine.train()

    # Sanity check state of the algorithm
    sanity_check_algorithm_state(ne_algorithm)


def test_codeepneat_2():
    # Create test config
    config = tfne.parse_configuration(os.path.dirname(__file__) + '/test_codeepneat_2_config.cfg')
    environment = tfne.environments.XOREnvironment(weight_training=True, config=config)
    ne_algorithm = tfne.algorithms.CoDeepNEAT(config)

    # Start test
    engine = tfne.EvolutionEngine(ne_algorithm=ne_algorithm,
                                  environment=environment,
                                  backup_dir_path=tempfile.gettempdir(),
                                  max_generations=6,
                                  max_fitness=None)
    engine.train()

    # Sanity check state of the algorithm
    sanity_check_algorithm_state(ne_algorithm)


def test_codeepneat_3():
    # Create test config
    config = tfne.parse_configuration(os.path.dirname(__file__) + '/test_codeepneat_3_config.cfg')
    environment = tfne.environments.XOREnvironment(weight_training=True, config=config)
    ne_algorithm = tfne.algorithms.CoDeepNEAT(config)

    # Start test
    engine = tfne.EvolutionEngine(ne_algorithm=ne_algorithm,
                                  environment=environment,
                                  backup_dir_path=tempfile.gettempdir(),
                                  max_generations=6,
                                  max_fitness=None)
    engine.train()

    # Sanity check state of the algorithm
    sanity_check_algorithm_state(ne_algorithm)


def test_codeepneat_4():
    # Create test config
    config = tfne.parse_configuration(os.path.dirname(__file__) + '/test_codeepneat_4_config.cfg')
    environment = tfne.environments.MNISTEnvironment(weight_training=True, config=config)
    ne_algorithm = tfne.algorithms.CoDeepNEAT(config)

    # Start test
    engine = tfne.EvolutionEngine(ne_algorithm=ne_algorithm,
                                  environment=environment,
                                  backup_dir_path=tempfile.gettempdir(),
                                  max_generations=2,
                                  max_fitness=None)
    engine.train()

    # Sanity check state of the algorithm
    sanity_check_algorithm_state(ne_algorithm)


def test_codeepneat_5():
    # Create test config
    config = tfne.parse_configuration(os.path.dirname(__file__) + '/test_codeepneat_4_config.cfg')
    environment = tfne.environments.CIFAR10Environment(weight_training=True, config=config)
    ne_algorithm = tfne.algorithms.CoDeepNEAT(config)

    # Start test
    engine = tfne.EvolutionEngine(ne_algorithm=ne_algorithm,
                                  environment=environment,
                                  backup_dir_path=tempfile.gettempdir(),
                                  max_generations=2,
                                  max_fitness=None)
    engine.train()

    # Sanity check state of the algorithm
    sanity_check_algorithm_state(ne_algorithm)
