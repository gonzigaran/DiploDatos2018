# Inspired by https://keon.io/deep-q-learning/

import random
import gym
import math
import numpy as np
from collections import deque
from sklearn.linear_model import SGDRegressor


class SGDMountainCarSolver:
    def __init__(self, n_episodes=1000, max_env_steps=None, gamma=0.9, epsilon=1.0, epsilon_min=0.01,
                 epsilon_decay=0.005, alpha=0.0001, batch_size=32, monitor=False, cutoff=2000):

        self.memory = deque(maxlen=100000)
        self.env = gym.make('MountainCar-v0')
        self.cutoff = cutoff
        self.env._max_episode_steps = self.cutoff

        if monitor:  # whether or not to display video
            self.env = gym.wrappers.Monitor(self.env, '../data/mountaincar-1', force=True)

        # hyper-parameter setting
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.alpha = alpha
        self.n_episodes = n_episodes
        self.batch_size = batch_size
        if max_env_steps is not None:
            self.env._max_episode_steps = max_env_steps

        # Init model
        self.model = SGDRegressor(
                alpha=self.alpha,
                learning_rate='optimal',
                shuffle=False,
                warm_start=True)

        # initialize model
        self.model.partial_fit(self.preprocess_state(self.env.reset(), 0), [0])

    def remember(self, state, action, reward, next_state, done):
        """In this method, the (s, a, r, s') tuple is stored in the memory"""
        self.memory.append((state, action, reward, next_state, done))

    def choose_action(self, state, epsilon):
        """Chooses the next action according to the model trained and the policy"""

        qsa = np.asarray([self.model.predict(self.preprocess_state(state, a))
                          for a in range(self.env.action_space.n)]).flatten()

        return self.env.action_space.sample() if (np.random.random() <= epsilon) \
            else np.argmax(qsa)  # exploits the current knowledge if the random number > epsilon, otherwise explores

    def get_epsilon(self, episode):
        """Returns an epsilon that decays over time until a minimum epsilon value is reached; in this case the minimum
        value is returned"""
        return 0.9

    @staticmethod
    def preprocess_state(state, action):
        """State and action are stacked horizontally to be passed as an input of the approximator"""
        return np.reshape(np.hstack((state, action)), [1, 3])

    def replay(self, batch_size):
        """Previously stored (s, a, r, s') tuples are replayed (that is, are added into the model). The size of the
        tuples added is determined by the batch_size parameter"""

        x_batch, y_batch = [], []
        minibatch = random.sample(self.memory, min(len(self.memory), batch_size))

        for state, action, reward, next_state, done in minibatch:
            qsa_s_prime = np.asarray([self.model.predict(self.preprocess_state(next_state, a))
                                      for a in range(self.env.action_space.n)])

            qsa_s = reward if done \
                else reward + self.gamma * np.max(qsa_s_prime)

            x_batch.append(self.preprocess_state(state, action)[0])
            y_batch.append(qsa_s)

        self.model.partial_fit(np.array(x_batch), np.array(y_batch))

    def run(self):
        """Main loop that controls the execution of the agent"""

        scores100 = deque(maxlen=100)
        scores = []
        for e in range(self.n_episodes):
            state = self.env.reset()
            done = False
            t = 0  # t counts the number of time-steps the pole has been kept up
            
            while not done:
                action = self.choose_action(state, self.get_epsilon(e))
                next_state, reward, done, _ = self.env.step(action)
                if state[1] > 0 and action == 2:
                    reward = 1
                if state[1] < 0 and action == 0:
                    reward = 1
                if state[0] >= 0.5:
                    reward = 1
                
                self.remember(state, action, reward, next_state, done)
                self.replay(self.batch_size)

                state = next_state
                t -= 1
            scores100.append(t)
            scores.append(t)
            mean_score = np.mean(scores100)
            if e % 100 == 0:
                print('[Episode {}] - Mean penalty over last 100 episodes was {}.'.format(e, mean_score))

        # noinspection PyUnboundLocalVariable
        print('[Episode {}] - Mean penalty over last 100 episodes was {}.'.format(e, mean_score))
        return scores


if __name__ == '__main__':
    """Code added in order to enable executing this solver as a standalone script"""
    agent = SGDMountainCarSolver()
    agent.run()
