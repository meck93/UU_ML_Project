import gym
import numpy as np
import retro
from baselines.common.atari_wrappers import FrameStack

import cv2
from config import HEIGHT, N_FRAMES, WIDTH  # hyperparameters

cv2.ocl.setUseOpenCL(False)


class PreprocessFrames(gym.ObservationWrapper):
    def __init__(self, env):
        """Preprocess and wrap frames to HEIGHTxWIDTH."""
        gym.ObservationWrapper.__init__(self, env)
        self.width = WIDTH
        self.height = HEIGHT
        self.observation_space = gym.spaces.Box(low=0, high=255, shape=(self.height, self.width, 1), dtype=np.uint8)

    def observation(self, frame):
        # transform color to grayscale
        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

        # crop the image top and bottom since it's static
        frame_cropped = frame_gray[9:-35, :]

        # resize the cropped image to WIDTHxHEIGHT
        frame = cv2.resize(frame_cropped, (self.width, self.height), interpolation=cv2.INTER_AREA)
        return frame[:, :, None]


class MarioDiscretizer(gym.ActionWrapper):
    """
    Wrap a gym-retro environment and make it use discrete
    actions for the SuperMarioBros game.
    """

    def __init__(self, env):
        super(MarioDiscretizer, self).__init__(env)
        # All buttons of the NES
        buttons = ['B', None, 'SELECT', 'START', 'UP', 'DOWN', 'LEFT', 'RIGHT', 'A']

        # Custom discrete actions defined by ourselves
        # Limits the number of possible actions and should improve training time
        actions = [[None], ['LEFT'], ['RIGHT'], ['RIGHT', 'A'], ['RIGHT', 'B'], ['RIGHT', 'A', 'B'], ['A']]
        # actions = [[None], ['LEFT'], ['RIGHT'], ['RIGHT', 'A'], ['A']]
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
    env = retro.make(game='SuperMarioBros3-Nes', state="1Player.World1.Level1.state",
                     scenario="./data/scenario.json", record="./recordings/V2/")

    if disc_acts:
        # Build the actions array
        env = MarioDiscretizer(env)

    # PreprocessFrame
    env = PreprocessFrames(env)

    # Stack N_FRAMES number of frames
    env = FrameStack(env, N_FRAMES)

    return env

# TODO: code that can be used to plot the preprocessing
# import matplotlib.pyplot as plt
# f, axs = plt.subplots(2, 2, figsize=(15, 15))
# axs[0, 0].set_title("Raw Input Image")
# axs[0, 0].imshow(frame)
# axs[0, 0].set_ylim((224, 0))
# axs[0, 0].set_yticks(np.arange(0, 225, 224//16))
# axs[0, 0].set_xlim((0, 240))
# axs[0, 0].set_xticks(np.arange(0, 241, 240//16))

# # transform color to grayscale
# frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)

# axs[0, 1].set_title("Gray-Scale Image")
# axs[0, 1].imshow(frame_gray, cmap="gray", vmin=0, vmax=255)
# axs[0, 1].set_ylim((224, 0))
# axs[0, 1].set_yticks(np.arange(0, 225, 224//16))
# axs[0, 1].set_xlim((0, 240))
# axs[0, 1].set_xticks(np.arange(0, 241, 240//16))

# # crop the image top and bottom since it's static
# frame_cropped = frame_gray[9:-35, :]

# axs[1, 0].set_title("Cropped Image")
# axs[1, 0].imshow(frame_cropped, cmap="gray", vmin=0, vmax=255)
# axs[1, 0].set_ylim((224, 0))
# axs[1, 0].set_yticks(np.arange(0, 225, 224//16))
# axs[1, 0].set_xlim((0, 240))
# axs[1, 0].set_xticks(np.arange(0, 241, 240//16))

# # normalize the values to range [0,1]
# frame_normalized = frame_cropped / 255.0

# # resize the cropped image to WIDTHxHEIGHT
# frame = cv2.resize(frame_normalized, (self.width, self.height), interpolation=cv2.INTER_AREA)

# axs[1, 1].set_title("Downsized Image")
# axs[1, 1].imshow(frame, cmap="gray", vmin=0, vmax=1)
# axs[1, 1].set_ylim((84, 0))
# axs[1, 1].set_yticks(np.arange(0, 85, 84//7))
# axs[1, 1].set_xlim((0, 84))
# axs[1, 1].set_xticks(np.arange(0, 85, 84//7))
# plt.show()
