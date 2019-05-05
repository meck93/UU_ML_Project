import tensorflow as tf
import numpy as np
import random
import gym
import retro

# hyper parameters
from config import *

# custom Mario environment
from environment import make_custom_env

# DeepQNetwork and Memory
from model import DQNetwork
from utils import ReplayMemory


class Agent:
    def __init__(self, level_name):
        # level name == model name => set in train.py/play.py
        self.level_name = level_name

        # setup environment
        self.env = make_custom_env(disc_acts=True)

        # one hot encoded version of our actions
        self.possible_actions = np.array(np.identity(self.env.action_space.n, dtype=int).tolist())

        # resest graph
        tf.reset_default_graph()

        # instantiate the DQNetwork
        self.DQNetwork = DQNetwork(state_size, self.env.action_space.n, learning_rate)

        # instantiate memory
        self.memory = ReplayMemory(max_size=memory_size)

        # saver will help us save our model
        self.saver = tf.train.Saver()

        # setup tensorboard writer
        self.writer = tf.summary.FileWriter("logs/{}/".format(self.level_name))

        # write initial loss
        tf.summary.scalar("Loss", self.DQNetwork.loss)
        self.write_op = tf.summary.merge_all()

        # initialize the memory
        for i in range(pretrain_length):
            # If it's the first step
            if i == 0:
                state = self.env.reset()

            # Get next state, the rewards, done by taking a random action
            choice = random.randint(1, len(self.possible_actions)) - 1
            action = self.possible_actions[choice]
            next_state, reward, done, info = self.env.step(choice)

            # set the initial number of lives
            self.lives = int(info['lives'])

            # if the episode is finished (we're dead)
            if done:
                # we inished the episode
                next_state = np.zeros((WIDTH, HEIGHT, N_FRAMES), dtype=np.int)
                print("Next State Shape:", next_state.shape)
                state = np.array(state)
                print("State Shape:", state.shape)

                # add experience to memory
                self.memory.add((state, action, reward, next_state, done))

                # start a new episode
                state = self.env.reset()
            else:
                # add experience to memory
                self.memory.add((state, action, reward, next_state, done))

                # our new state is now the next_state
                state = next_state

    def predict_action(self, sess, explore_start, explore_stop, decay_rate, decay_step, state, actions):
        # first we randomize a number
        exp_exp_tradeoff = np.random.rand()

        # compute the current exploration probability
        explore_probability = explore_stop + (explore_start - explore_stop) * np.exp(-decay_rate * decay_step)

        if explore_probability > exp_exp_tradeoff:
            # make a random action
            choice = random.randint(1, len(self.possible_actions)) - 1
            action = self.possible_actions[choice]
        else:
            # transform LazyFrames into np array [None, HEIGHT, WIDTH, N_FRAMES]
            state = np.array(state)

            # estimate the Qs values state
            Qs = sess.run(self.DQNetwork.output, feed_dict={self.DQNetwork.inputs_: state.reshape((1, *state.shape))})

            # take the biggest Q value (= best action)
            choice = np.argmax(Qs)
            action = self.possible_actions[choice]

        return action, choice, explore_probability

    def play(self):
        with tf.Session() as sess:
            total_test_rewards = []

            # Load the model
            self.saver.restore(sess, "models/{0}.cpkt".format(self.level_name))

            for episode in range(4):
                total_rewards = 0

                state = self.env.reset()

                print("****************************************************")
                print("EPISODE ", episode)

                while True:
                    # transform LazyFrames into np array [None, HEIGHT, WIDTH, N_FRAMES]
                    state = np.array(state)

                    # Reshape the state
                    state = state.reshape((1, *state_size))

                    # Get action from Q-network: estimate the Qs values state
                    Qs = sess.run(self.DQNetwork.output, feed_dict={self.DQNetwork.inputs_: state})

                    # Take the biggest Q value (= the best action)
                    choice = np.argmax(Qs)

                    # Perform the action and get the next_state, reward, and done information
                    next_state, reward, done, info = self.env.step(choice)
                    self.env.render()

                    total_rewards += reward

                    if done or int(info['lives']) != self.lives:
                        print("Score", total_rewards)
                        total_test_rewards.append(total_rewards)
                        break

                    state = next_state
            self.env.close()

    def train(self):
        with tf.Session() as sess:
            # initialize the variables
            sess.run(tf.global_variables_initializer())

            # initialize decay rate (that will be used to reduce epsilon)
            decay_step = 0

            for episode in range(TOTAL_EPISODES):
                # set step to 0
                step = 0

                # initialize the x_position to 0
                x_pos_tracker = 0

                # initialize killed and stuck to False
                killed = False
                stuck = False

                # initialize rewards of episode
                episode_rewards = []

                # make a new episode and observe the first state
                state = self.env.reset()

                print("Episode:", episode)

                while step < MAX_STEPS:
                    step += 1

                    # increase decay_step
                    decay_step += 1

                    # predict an action
                    action, choice, explore_probability = self.predict_action(
                        sess, explore_start, explore_stop, decay_rate, decay_step, state, self.possible_actions)

                    # perform the action and get the next_state, reward, and done information
                    next_state, reward, done, info = self.env.step(choice)

                    # add the reward to total reward
                    episode_rewards.append(reward)

                    if episode_render:
                        self.env.render()

                    # check the x_position of Mario every 500 steps to see if he's stuck
                    if step % STUCK_STEPS == 0:
                        # compute the current x_position of Mario
                        # x_pos + 255*xpos_multiplier since x_pos only goes from 0 to 255
                        new_x_pos = int(info['xpos']) + int(info['xpos_multiplier'])*255

                        if new_x_pos == x_pos_tracker:
                            print("Mario is stuck! Restarting!")
                            stuck = True
                        else:
                            x_pos_tracker = new_x_pos

                    # check if Mario is still alive
                    if int(info['lives']) != self.lives:
                        print("Mario has died! Restarting!")
                        killed = True

                    # check if Mario has finished the level
                    if done:
                        print("Episode ended!")

                    if done or killed or stuck:
                        # the episode ends so no next state
                        next_state = np.zeros((WIDTH, HEIGHT, N_FRAMES), dtype=np.int)

                        # set step = MAX_STEPS to end episode
                        step = MAX_STEPS

                        # get total reward of the episode
                        total_reward = np.sum(episode_rewards)

                        print("Episode:", episode, "Total reward:", total_reward,
                              "Explore P:", explore_probability, "Training Loss:", loss)

                        # rewards_list.append((episode, total_reward))

                        # store transition <s_i, a, r_{i+1}, s_{i+1}> in memory
                        self.memory.add((state, action, reward, next_state, done))
                    else:
                        # store transition <s_i, a, r_{i+1}, s_{i+1}> in memory
                        self.memory.add((state, action, reward, next_state, done))

                        # s_{i} := s_{i+1}
                        state = next_state

                    #### LEARNING PART ####
                    # Obtain random mini-batch from memory
                    states_t, actions, rewards, states_tp1, dones = self.memory.sample(batch_size)

                    target_Qs_batch = []

                    # get Q values for the states_tp1 (next states)
                    Qs_next_state = sess.run(self.DQNetwork.output, feed_dict={self.DQNetwork.inputs_: states_tp1})

                    # set Q_target = r if episode ends with s+1
                    for i in range(batch_size):
                        terminal = dones[i]

                    # if we are in a terminal state, only equals reward
                    if terminal:
                        target_Qs_batch.append(rewards[i])
                    else:
                        target = rewards[i] + gamma * np.max(Qs_next_state[i])
                        target_Qs_batch.append(target)

                    targets_mb = np.array([each for each in target_Qs_batch])

                    loss, _ = sess.run([self.DQNetwork.loss, self.DQNetwork.optimizer], feed_dict={self.DQNetwork.inputs_: states_t,
                                                                                                   self.DQNetwork.target_Q: targets_mb,
                                                                                                   self.DQNetwork.actions_: actions})

                    # write tf summaries
                    summary = sess.run(self.write_op, feed_dict={self.DQNetwork.inputs_: states_t,
                                                                 self.DQNetwork.target_Q: targets_mb,
                                                                 self.DQNetwork.actions_: actions})
                    self.writer.add_summary(summary, episode)
                    self.writer.flush()

                # save model every 5 episodes
                if episode % 5 == 0:
                    self.saver.save(sess, "models/{0}.cpkt".format(self.level_name))
                    print("Model Saved")
