import numpy as np
import tensorflow as tf

import neuroevolution as ne


def test_xor():
    assert tf.__version__ == "2.0.0-beta1"

    genome_prototype = {
        0: {
            'inputs': [1, 4],
            'outputs': [2],
            'layer_activations': [tf.keras.activations.tanh],
            'out_activation': tf.keras.activations.sigmoid,
            'default_activation': tf.keras.activations.sigmoid
        },
        1: (1, 2),
        2: (1, 3),
        3: (4, 3),
        4: (4, 2),
        5: (3, 2),
    }
    print("Custom Genotype: {}".format(genome_prototype))

    genome = ne.encodings.direct.DirectEncodingGenome(1, genome_prototype, check_genome_sanity=True)
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
