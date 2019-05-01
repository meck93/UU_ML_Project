import retro  # pip install gym-retro
import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
import neat  # pip install neat-python
import pickle  # pip install cloudpickle

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT,COMPLEX_MOVEMENT
#/Users/andrastefanianegus/miniconda3/envs/supermario/lib/python3.7/site-packages/gym_super_mario_bros-7.1.6.dist-info
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

t=pickle.load(open('winner.pkl', 'rb'))
print(t)