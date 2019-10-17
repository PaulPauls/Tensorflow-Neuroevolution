import gym
import numpy as np

from ..base_environment import BaseEnvironment


class CartPoleEnvironment(BaseEnvironment):
    """
    Environment for the TFNE framework that represents the simulation of the CartPole environment by providing an
    interface to OpenAI Gym's CartPole environment. This environment does not train the weights of the supplied genomes
    and their phenotype models in any way.
    """

    def __init__(self, render_flag):
        self.env = gym.make("CartPole-v1")
        self.render_flag = render_flag

    def eval_genome_fitness(self, genome) -> float:
        """
        Calculate and return the genome fitness as the percentage of the maximum steps (500) the genome's model
        perservered in doing the CartPole task and balancing it out.
        :param genome: genome of the TFNE framework, providing a built Tensorflow model
        :return: genome model accuracy as percentage of the maximum steps (500) achieved in balancing out the CartPole
        """
        model = genome.get_model()
        total_reward = 0
        done = False

        state = self.env.reset()
        state = np.reshape(state, [1, state.shape[0]])
        while not done:
            if self.render_flag:
                self.env.render()
            model_prediction = model.predict(state)
            action = np.argmax(model_prediction[0])
            state, reward, done, _ = self.env.step(action)
            state = np.reshape(state, [1, state.shape[0]])
            total_reward += reward

        self.env.close()
        # Divide total reward by 5.0 as the maximum total_reward is 500 (as 500 is maximum steps in environment,
        # dictated by OpenAI Gym).
        return total_reward / 5.0

    def replay_genome(self, genome):
        """
        Replay the genome by applying the model outputs to the environment while rendering it all the while, visually
        replaying the genomes ability to solve the environment.
        :param genome: genome of the TFNE framework, providing a built Tensorflow model
        """
        model = genome.get_model()
        done = False

        state = self.env.reset()
        state = np.reshape(state, [1, state.shape[0]])
        while not done:
            self.env.render()
            model_prediction = model.predict(state)
            action = np.argmax(model_prediction[0])
            state, _, done, _ = self.env.step(action)
            state = np.reshape(state, [1, state.shape[0]])

        self.env.close()

    def get_input_shape(self) -> ():
        """
        :return: one-dimensional tuple specifying the number of inputs for a model supplied to this environment
        """
        return self.env.observation_space.shape

    def get_num_output(self) -> int:
        return self.env.action_space.n
