import sys
import ast
from typing import Union, Any
from configparser import ConfigParser

import tensorflow as tf
from PyQt5 import QtWidgets

from tfne.visualizer import TFNEVWelcomeWindow


def parse_configuration(config_path) -> object:
    """
    Takes a configuration file path, reads the configuration in ConfigParser and then returns it.
    @param config_path: string of configuration file path
    @return: ConfigParser instance
    """
    config = ConfigParser()
    with open(config_path) as config_file:
        config.read_file(config_file)

    return config


def read_option_from_config(config, section, option) -> Any:
    """
    @param config: ConfigParser instance
    @param section: string of the config section to read from
    @param option: string of the config section option to read
    @return: literal evaluated value of the config option
    """
    value = ast.literal_eval(config[section][option])
    print("Config value for '{}/{}': {}".format(section, option, value))
    return value


def round_with_step(value, minimum, maximum, step) -> Union[int, float]:
    """
    @param value: int or float value to round
    @param minimum: int or float specifying the minimum value the rounded result can take
    @param maximum: int or float specifying the maximum value the rounded result can take
    @param step: int or float step value of which the rounded result has to be a multiple of.
    @return: Rounded int or float (identical type to 'value') that is a multiple of the supplied step
    """
    lower_step = int(value / step) * step
    if value % step - (step / 2.0) < 0:
        if minimum <= lower_step <= maximum:
            return lower_step
        if lower_step < minimum:
            return minimum
        if lower_step > maximum:
            return maximum
    else:
        higher_step = lower_step + step
        if minimum <= higher_step <= maximum:
            return higher_step
        if higher_step < minimum:
            return minimum
        if higher_step > maximum:
            return maximum


def start_visualizer(tfne_state_backup_dir_path=None):
    """
    Starts TFNE visualizer and optionally takes the directory path of the TFNE state backup
    @param tfne_state_backup_dir_path: Optional string specifying the directory of the TFNE state backup
    """
    tfnev = QtWidgets.QApplication(sys.argv)
    tfnev_welcomewindow = TFNEVWelcomeWindow(tfne_state_backup_dir_path)
    tfnev.exec()


def set_tensorflow_memory_growth():
    """
    Set memory growth to true in the Tensorflow backend, fixing memory allocation problems that can occur on NVIDIA
    Turing GPUs
    """
    gpus = tf.config.experimental.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
