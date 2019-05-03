import environment

from keras.models import Sequential
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
    def __init__(self, state_size, action_size):
        self.state_size = state_size # TODO: Can possibly be removed
        self.action_size = action_size
        self.memory = deque(maxlen=1000000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
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
        model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

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

    # TODO: Reshape the state array to the correct dimensions
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

# TODO: remove magical numbers, preprocess_frame size
# TODO: Write function to load the trained model and test it.
def main():
    total_episodes = 50
    max_steps = 50000
    batch_size = 16
    episode_render = True

    env = environment.make_custom_env(disc_acts=True)
    agent = DQNAgent([96,96,4], env.action_space.n)


    for episode in range(total_episodes):
        step = 0
        episode_rewards = []
        state = env.reset()

        while step < max_steps:
            step += 1

            action = agent.act(state)
            next_state, reward, done, info = env.step(action)
            episode_rewards.append(reward)

            if episode_render:
                env.render()

            if done:
                step = max_steps

                total_reward = np.sum(episode_rewards)

                print("Episode: {}/{}".format(episode, total_episodes),
                      "Total reward: {}".format(total_reward))
            else:
                agent.remember(state, action, reward, next_state, done)
                state = next_state

            if len(agent.memory) > batch_size:
                agent.replay(batch_size)

    # TODO: Save model every 10 episodes
    if episode % 10 == 0:
        print("Model saved")

if __name__ == "__main__":
    main()

