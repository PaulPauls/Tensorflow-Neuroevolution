import random
import gym
import numpy as np
import tensorflow as tf

from collections import deque
import neuroevolution as ne


def test_cartpole():
    assert tf.__version__ == "2.0.0-beta1"

    genome_prototype = {
        0: {
            'inputs': [1, 2, 3, 4],
            'outputs': [5, 6],
            'layer_activations': [tf.keras.activations.tanh],
            'out_activation': tf.keras.activations.sigmoid,
            'default_activation': tf.keras.activations.softmax
        },
        1: (1, 5),
        2: (1, 6),
        3: (2, 5),
        4: (2, 6),
        5: (3, 5),
        6: (3, 6),
        7: (4, 5),
        8: (4, 6)
    }
    print("Custom Genotype: {}".format(genome_prototype))

    genome = ne.encodings.direct.DirectEncodingGenome(1, genome_prototype, check_genome_sanity=True)
    genome.summary()
    genome.visualize()

    if True:
        env = gym.make("CartPole-v0")
        input_shape = env.observation_space.shape
        num_output = env.action_space.n
        print("Input Shape: {} \tOutput Shape: {}".format(input_shape, num_output))

        model = genome.get_phenotype_model()
        model.compile(
            optimizer=tf.keras.optimizers.Adam(lr=0.01),
            loss=tf.keras.losses.MeanSquaredError()
        )

        max_reward = 200
        batch_size = 64
        memory_size = batch_size * 10
        memory = deque(maxlen=memory_size)
        review_after_n_runs = 10

        max_runs = 1000
        run_counter = 0
        while True:  # run_counter < max_runs:
            run_memory = []
            total_reward = 0
            total_reward_memory = deque(maxlen=review_after_n_runs)
            done = False
            step = 0

            state = env.reset()
            state = np.reshape(state, [1, 4])
            while not done:
                env.render()
                model_prediction = model.predict(state)
                action = np.argmax(model_prediction[0])
                state, reward, done, _ = env.step(action)
                state = np.reshape(state, [1, 4])
                total_reward += reward if not done else -reward
                run_memory.append((state, model_prediction, action))
                total_reward_memory.append(total_reward)
                step += 1

            for (state, model_prediction, action) in run_memory:
                memory.append((state, model_prediction, action, total_reward))

            if len(memory) > batch_size:
                train_model_on_batch(model, memory, batch_size, max_reward)
            if run_counter % review_after_n_runs == 0 and run_counter != 0:
                avg_total_reward = np.mean(total_reward_memory)
                print('Run Count: {} \t Average reward over last {} runs: {}'.
                      format(run_counter, review_after_n_runs, avg_total_reward))
            run_counter += 1

    print("FIN")


def train_model_on_batch(model, memory, batch_size, max_reward):
    batch = random.sample(memory, batch_size)

    x_batch = None
    y_batch = None

    for (state, model_prediction, action, total_reward) in batch:

        if state[0, 2] < 0:
            adjusted_model_prediction = np.array([[1, 0]])
        else:
            adjusted_model_prediction = np.array([[0, 1]])
        '''
        adjusted_model_prediction = model_prediction.copy()
        adjusted_model_prediction[0, action] += total_reward/max_reward
        '''

        x_batch = state if x_batch is None else tf.keras.layers.concatenate([x_batch, state], axis=0)
        y_batch = adjusted_model_prediction if y_batch is None else \
            tf.keras.layers.concatenate([y_batch, adjusted_model_prediction], axis=0)

    model.fit(x_batch, y_batch, verbose=0)


if __name__ == '__main__':
    test_cartpole()
