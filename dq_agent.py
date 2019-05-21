import environment

from keras.models import Sequential, load_model
from keras.layers import Conv2D, Dense, Flatten
from keras.initializers import Orthogonal
from keras.optimizers import Adam
from keras import backend as K

import retro
import gym

import numpy as np

import random

from collections import deque

class DQNAgent:
    def __init__(self, action_size):
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.6
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.9999
        self.learning_rate = 0.001
        self.model = self._build_model()

    def _build_model(self):
        model = Sequential()
        model.add(self.conv_layer(32, 8, 4))
        model.add(self.conv_layer(64, 4, 2))
        model.add(self.conv_layer(64, 3, 2))
        model.add(Flatten())
        model.add(self.fc_layer(512))
        model.add(self.fc_layer(self.action_size, None))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))

        return model

    @staticmethod
    def conv_layer(filters, kernel_size, strides):
        return Conv2D(filters=filters,
                      kernel_size=kernel_size,
                      strides=(strides, strides),
                      activation='relu',
                      kernel_initializer=Orthogonal())
    @staticmethod
    def fc_layer(units, activation_fn='relu'):
        return Dense(units=units,
                     activation=activation_fn,
                     kernel_initializer=Orthogonal())

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        else:
            act_values = self.model.predict(np.expand_dims(np.array(state), axis=0))
            return np.argmax(act_values[0])

    def act_test(self, state):
        act_values = self.model.predict(np.expand_dims(np.array(state), axis=0))
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = self.model.predict(np.expand_dims(np.array(state), axis=0))

            if done:
                target[0][action] = reward
            else:
                t = self.model.predict(np.expand_dims(np.array(next_state), axis=0))[0]
                target[0][action] = reward + self.gamma * np.amax(t)

            self.model.fit(np.expand_dims(np.array(state), axis=0), target, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def save_model(self, file_name):
        print("Model saved")
        self.model.save("trained_models/" + file_name + ".h5")

    def load_model(self, file_name):
        print("Model loaded")
        self.model = load_model("trained_models/" + file_name + ".h5")

class Reward:
    def __init__(self):
        self.xpos = 24
        self.time = 298

    def calculate(self, info):
        new_xpos = info['xpos'] + 255 * info['xpos_multiplier']
        x_pos_rew, self.xpos = new_xpos - self.xpos, new_xpos
        time_rew, self.time = self.time - info['time'], info['time']
        death_rew = 0 if info['lives'] > 3 else -15
        score_rew = info['score'] / 10
        reward = x_pos_rew + time_rew + death_rew + score_rew

        if reward > 15:
            reward = 15
        elif reward < -15:
            reward = -15

        return reward

    def reset(self):
        self.xpos = 24
        self.time = 298

def test(model_name, total_episodes, episode_render):
    env = environment.make_custom_env(disc_acts=True)
    agent = DQNAgent(env.action_space.n)
    agent.load_model(model_name)
    rew = Reward()

    for episode in range(total_episodes):
        episode_rewards = []
        done = False
        state = env.reset()
        rew.reset()

        while not done:
            action = agent.act_test(state)
            next_state, reward, done, info = env.step(action)
            reward = rew.calculate(info)
            episode_rewards.append(reward)

            if episode_render:
                env.render()

            state = next_state

        total_reward = np.sum(episode_rewards)

        print("Episode: {}/{}".format(episode, total_episodes),
              "Total reward: {}".format(total_reward))

def train(model_name, total_episodes=100, max_steps=500, batch_size=8, episode_render=True):
    env = environment.make_custom_env(disc_acts=True)
    agent = DQNAgent(env.action_space.n)
    rew = Reward()

    for episode in range(total_episodes):
        step = 0
        episode_rewards = []
        state = env.reset()
        rew.reset()

        while step < max_steps:
            step += 1
            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            reward = rew.calculate(info)
            episode_rewards.append(reward)

            if episode_render:
                env.render()

            if done or info['lives'] < 4 or step >= max_steps:
                step = max_steps
                total_reward = np.sum(episode_rewards)

                print("Episode: {}/{}".format(episode, total_episodes),
                      "Total reward: {}".format(total_reward),
                      "Exploration rate: {}".format(agent.epsilon))
                agent.remember(state, action, reward, next_state, done)
            else:
                agent.remember(state, action, reward, next_state, done)
                state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

        if episode % 1 == 0:
            max_steps += 100
            agent.save_model(model_name)

#test("dq_agent_1", total_episodes=10000, episode_render=True)
train("dq_agent_1", total_episodes=10000, max_steps=500, batch_size=16, episode_render=True)