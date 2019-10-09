import os
from configparser import ConfigParser


def load_config(config_path) -> ConfigParser:
    """
    Loading configuration from supplied path and processing it with a ConfigParser, which is then returned.
    :param config_path: str path to the configuration file to be read
    :return: ConfigParser Object which has processed the supplied configuration
    """
    if not os.path.isfile(config_path):
        raise FileNotFoundError("Specified configuration file does not exist: " + os.path.abspath(config_path))

    config = ConfigParser()
    with open(config_path) as config_file:
        config.read_file(config_file)

    return config
