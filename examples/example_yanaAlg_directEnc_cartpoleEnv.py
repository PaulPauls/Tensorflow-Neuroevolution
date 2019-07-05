import tensorflow as tf

import neuroevolution as ne


def main():
    """
    A simple example used in the current alpha stage of development to show of the Tensorflow-Neuroevolution framework.
    This example uses the YANA ne-algorithm with a direct encoded genome to solve the OpenAI gym cartpole environment.

    :return: None
    """

    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
    logger = tf.get_logger()

    logger.debug(tf.__version__)

    config = ne.load_config('./example_yanaAlg_directEnc_cartpoleEnv.cfg')


if __name__ == '__main__':
    main()
