import numpy as np
import random
from collections import deque  # ordererd collection with ends


# memory for the agent to remember its experiences
class ReplayMemory:
    def __init__(self, max_size):
        self._buffer = deque(maxlen=max_size)

    def __len__(self):
        return len(self._buffer)

    def add(self, experience):
        self._buffer.append(experience)

    def sample(self, batch_size):
        """Sample a batch of experiences.

        Parameters
        ----------
        batch_size: int
            How many transitions to sample.

        Returns
        -------
        states_t: np.array
            batch of states
        act_batch: np.array
            batch of actions executed given obs_batch
        rew_batch: np.array
            rewards received as results of executing act_batch
        states_tp1_batch: np.array
            next set of states seen after executing act_batch
        done_mask: np.array
            done_mask[i] = 1 if executing act_batch[i] resulted in
            the end of an episode and 0 otherwise.
        """
        idxes = [random.randint(0, len(self._buffer) - 1) for _ in range(batch_size)]
        return self._encode_samples(idxes)

    def _encode_samples(self, idxes):
        states_t, actions, rewards, states_tp1, dones = [], [], [], [], []

        for i in idxes:
            experience = self._buffer[i]
            state_t, action, reward, state_tp1, done = experience

            states_t.append(np.array(state_t, copy=False))
            actions.append(np.array(action, copy=False))
            rewards.append(reward)
            states_tp1.append(np.array(state_tp1, copy=False))
            dones.append(done)

        return np.array(states_t), np.array(actions), np.array(rewards), np.array(states_tp1), np.array(dones)
