#!/usr/bin/env python

import numpy as np
import gym
import retro

# This will be useful for stacking frames
from baselines.common.atari_wrappers import FrameStack

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
                   ['RIGHT', 'B'], ['RIGHT', 'A', 'B'], ['A'], ['A', 'A']]
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

    return env
