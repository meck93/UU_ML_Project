#!/usr/bin/env python

import numpy as np
import gym
import retro


from collections import deque

import cv2
cv2.ocl.setUseOpenCL(False)


class PreprocessFrames(gym.ObservationWrapper):
    def __init__(self, env):
        """Warp frames to 96x96."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = 96
        self.height = 96
        self.observation_space = gym.spaces.Box(low=0, high=255,
                                                shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        frame = cv2.resize(frame, (self.width, self.height),
                           interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class MarioDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the SuperMarioBros game.
    """

    def __init__(self, env):
        super(MarioDiscretizer, self).__init__(env)
        # All buttons of the NES
        buttons = ['B', None, 'SELECT', 'START',
                   'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

        # Custom discrete actions defined by ourselves
        # Limits the number of possible actions and should improve training time
        actions = [[None], ['LEFT'], ['RIGHT'], ['RIGHT', 'A'],
                   ['RIGHT', 'B'], ['RIGHT', 'A', 'B'], ['A'], ['UP']]
        self._actions = []

        for action in actions:
            arr = np.array([False] * len(buttons))
            for button in action:
                arr[buttons.index(button)] = True
            self._actions.append(arr)

        # maps each action to a discrete number
        self.action_space = gym.spaces.Discrete(len(self._actions))

    def action(self, a):  # pylint: disable=W0221
        return self._actions[a].copy()


class AllowBacktracking(gym.Wrapper):
    """
    Use deltas in max(X) as the reward, rather than deltas
    in X. This way, agents are not discouraged too heavily
    from exploring backwards if there is no way to advance
    head-on in the level.
    """

    def __init__(self, env):
        super(AllowBacktracking, self).__init__(env)
        self._cur_x = 0
        self._max_x = 0

    def reset(self, **kwargs):  # pylint: disable=E0202
        self._cur_x = 0
        self._max_x = 0
        return self.env.reset(**kwargs)

    def step(self, action):  # pylint: disable=E0202
        obs, rew, done, info = self.env.step(action)
        self._cur_x += rew
        rew = max(0, self._cur_x - self._max_x)
        self._max_x = max(self._max_x, self._cur_x)
        return obs, rew, done, info


class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        """Stack k last frames. Returns lazy array, which is much more memory efficient. See Also
        baselines.common.atari_wrappers.LazyFrames
        """
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(
            shp[:-1] + (shp[-1] * k,)), dtype=env.observation_space.dtype)

    def reset(self):  # pylint: disable=E0202
        ob = self.env.reset()
        for _ in range(self.k):
            self.frames.append(ob)
        return self._get_ob()

    def step(self, action):  # pylint: disable=E0202
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._get_ob(), reward, done, info

    def _get_ob(self):
        assert len(self.frames) == self.k
        return LazyFrames(list(self.frames))


class LazyFrames(object):
    def __init__(self, frames):
        """This object ensures that common frames between the observations are only stored once.
        It exists purely to optimize memory usage which can be huge for DQN's 1M frames replay
        buffers.
        This object should only be converted to numpy array before being passed to the model.
        You'd not believe how complex the previous solution was."""
        self._frames = frames
        self._out = None

    def _force(self):
        if self._out is None:
            self._out = np.concatenate(self._frames, axis=-1)
            self._frames = None
        return self._out

    def __array__(self, dtype=None):
        out = self._force()
        if dtype is not None:
            out = out.astype(dtype)
        return out

    def __len__(self):
        return len(self._force())

    def __getitem__(self, i):
        return self._force()[..., i]


def make_custom_env(disc_acts=True):
    """
    Create an environment with some standard wrappers.
    """
    env = retro.make(game='SuperMarioBros3-Nes',
                     state="1Player.World1.Level1.state")

    if disc_acts:
        # Build the actions array
        env = MarioDiscretizer(env)

    # PreprocessFrame
    env = PreprocessFrames(env)

    # Stack 4 frames
    env = FrameStack(env, 4)

    # Allow back tracking that helps agents are not discouraged too heavily
    # from exploring backwards if there is no way to advance head-on in the level.
    env = AllowBacktracking(env)
    return env
