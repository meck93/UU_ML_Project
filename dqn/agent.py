import tensorflow as tf
import numpy as np
import random
import gym
import retro

# openai baselines
from baselines.common.schedules import LinearSchedule
from baselines.deepq.replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

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
        self.saver = tf.train.Saver(save_relative_paths=True)

        # setup tensorboard writer
        self.writer = tf.summary.FileWriter("logs/{}/".format(self.level_name))

        # write initial loss
        tf.summary.scalar("Loss", self.DQNetwork.loss)
        self.write_op = tf.summary.merge_all()

        # set the initial number of lives
        self.lives = 4

        # initialize the memory
        for i in range(pretrain_length):
            if i == 0:
                print("Initializing Memory with {} experiences!".format(pretrain_length))
                # initialize the x0 - previous position - to 24 (initial position)
                x0 = 24

                # reset the environment
                state = self.env.reset()

            # Get next state, the rewards, done by taking a random action
            choice = random.randint(1, len(self.possible_actions)) - 1
            action = self.possible_actions[choice]
            next_state, reward, done, info = self.env.step(choice)

            # compute the current x_position
            x1 = self._current_xpos(int(info['xpos']), int(info['xpos_multiplier']))

            # compute the positional reward
            x0, reward = self.x_pos_reward(x1, x0, reward)

            # check if Mario is still alive
            killed, reward = self.check_killed(int(info['lives']), reward)

            if done or killed:
                # we inished the episode
                next_state = np.zeros((WIDTH, HEIGHT, N_FRAMES), dtype=np.int)

                # add experience to memory
                self.memory.add((state, action, reward, next_state, done))

                # start a new episode
                state = self.env.reset()

                # reset x0 - previous position - to 24 (initial position)
                x0 = 24
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
            path = "./models/{}/".format(self.level_name)
            self.saver = tf.train.import_meta_graph("{}-4.meta".format(path))
            self.saver.restore(sess, tf.train.latest_checkpoint(path))

            for episode in range(1):
                total_rewards = 0
                step = 0

                state = self.env.reset()

                print("****************************************************")
                print("EPISODE ", episode)

                while step < MAX_STEPS:
                    step += 1

                    # transform LazyFrames into np array [None, HEIGHT, WIDTH, N_FRAMES]
                    state = np.array(state)

                    # Get action from Q-network: estimate the Qs values state
                    Qs = sess.run(self.DQNetwork.output, feed_dict={
                                  self.DQNetwork.inputs_: state.reshape((1, *state_size))})

                    # Take the biggest Q value (= the best action)
                    choice = np.argmax(Qs)
                    print(step, choice, Qs)

                    # Perform the action and get the next_state, reward, and done information
                    next_state, reward, done, info = self.env.step(choice)

                    # render the current state
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

                # initialize the stuck_pos_cp to 24 (initial position)
                stuck_pos_cp = 24

                # initialize the x0 - previous position - to 24 (initial position)
                x0 = 24

                # initialize stuck to False
                stuck = False

                # initialize rewards of episode
                episode_rewards = []

                # initialize episode loss
                episode_loss = []

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

                    if episode_render:
                        self.env.render()

                    print(info['time'])

                    # compute the current x_position
                    x1 = self._current_xpos(int(info['xpos']), int(info['xpos_multiplier']))

                    # compute the positional reward
                    x0, reward = self.x_pos_reward(x1, x0, reward)

                    # check if Mario is stuck
                    if step % STUCK_STEPS == 0:
                        stuck, reward = self.check_stuck(x1, stuck_pos_cp, reward)

                    # check if Mario is still alive
                    killed, reward = self.check_killed(int(info['lives']), reward)

                    # check if Mario has finished the level
                    if done:
                        print("\tEpisode ended!")

                    # add the reward to total reward
                    episode_rewards.append(reward)

                    if killed or stuck or done or step == MAX_STEPS:
                        # the episode ends so no next state
                        next_state = np.zeros((WIDTH, HEIGHT, N_FRAMES), dtype=np.int)

                        # set step = MAX_STEPS to end episode
                        step = MAX_STEPS

                        # get total reward of the episode
                        total_reward = np.sum(episode_rewards)
                        average_loss = np.mean(episode_loss)

                        print("Episode:", episode, "Total reward:", total_reward,
                              "Explore P:", explore_probability, "Average Training Loss:", average_loss)

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

                    # store loss
                    episode_loss.append(loss)

                    # write tf summaries
                    summary = sess.run(self.write_op, feed_dict={self.DQNetwork.inputs_: states_t,
                                                                 self.DQNetwork.target_Q: targets_mb,
                                                                 self.DQNetwork.actions_: actions})
                    self.writer.add_summary(summary, episode)
                    self.writer.flush()

                # save model every 5 episodes
                if episode % 5 == 0:
                    self.saver.save(sess, "./models/{0}/".format(self.level_name), global_step=episode)
                    print("Model Saved")

    def _current_xpos(self, xpos, xpos_multiplier):
        """
        Compute the current position of Mario.

        Inputs:
        - xpos: x_position (from 0 to 255)
        - xpos_multiplier: how many times the xpos has been looped 

        Returns:
        - current x_position
        """
        return xpos + xpos_multiplier*255

    def x_pos_reward(self, x1, x0, reward):
        """
        Computes the positional reward; reward = x1 - x0
        - x1: current position
        - x0: previous position

        Returns:
        - new previous position x0 = x1
        - update reward
        """
        reward = x1 - x0
        return x1, reward

    def check_stuck(self, xpos, stuck_pos_cp, reward):
        """
        Checks if Mario is stuck i.e. Mario's xpos has not changed since the last check.

        Inputs:
        - xpos: Mario's current x_position
        - stuck_pos_cp: Mario's x_position at the last check.
        - reward: the current step's reward

        Returns: 
        - stuck: bool - True if Mario's x_position hasn't changed
        - reward: float - updated reward
        """
        stuck = False
        if xpos == stuck_pos_cp:
            reward = 0
            stuck = True
            print("\tMario is stuck! Restarting!", reward)

        return stuck, reward

    def check_killed(self, curr_n_lives, reward):
        """        
        Checks if Mario has died. If so adjusts the reward.

        Inputs:
        - curr_n_lives: Mario's current number of lives
        - reward: the current step's reward

        Returns: 
        - killed: bool - True if Mario's has died.
        - reward: float - updated reward
        """
        killed = False
        if curr_n_lives != self.lives:
            reward = PENALTY_DYING  # update reward with dying penalty
            killed = True
            print("\tMario died! Restarting!", reward)

        return killed, reward
