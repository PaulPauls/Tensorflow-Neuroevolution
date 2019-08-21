import numpy as np
import tensorflow as tf

import neuroevolution as ne


def test_xor():
    # Assert that TF 2.x is used
    assert tf.__version__[0] == "2"

    genotype = {
        1: (1, 2),
        2: (1, 3),
        3: (4, 3),
        4: (4, 2),
        6: (1, 5),
        7: (4, 5),
        8: (5, 2),
        5: (3, 2)}

    activations = {'out_activation': tf.keras.activations.sigmoid,
                   'default_activation': tf.keras.activations.tanh}

    config = ne.load_config('../examples/xor_example/example_yanaAlg_directEnc_xorEnv.cfg')
    encoding = ne.encodings.DirectEncoding(config)

    genome = encoding.create_new_genome(genotype, activations, check_genome_sanity=True)
    genome.summary()
    genome.visualize()

    if True:
        x = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
        y = np.array([[0], [1], [1], [0]])

        model = genome.get_phenotype_model()

        model.compile(optimizer=tf.keras.optimizers.SGD(lr=0.1),
                      loss='binary_crossentropy')

        model.fit(x, y, batch_size=1, epochs=1000, verbose=1)

        print(model.summary())
        print(model.predict(x))

    print("FIN")


if __name__ == '__main__':
    test_xor()
