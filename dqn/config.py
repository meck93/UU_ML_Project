# current session
MODEL_NAME = "Test2"

# model hyperparameters
HEIGHT = 96
WIDTH = 96
N_FRAMES = 4
state_size = [HEIGHT, WIDTH, N_FRAMES]  # 4 stacked frames

# training hyperparameters
learning_rate = 0.00025  # alpha (aka learning rate)
TOTAL_EPISODES = 100  # total episodes for training
MAX_STEPS = 1500  # max possible steps in an episode
STUCK_STEPS = 500  # max steps in the same x_position until Mario is considered stuck
batch_size = 32

# exploration parameters
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.00001  # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.95  # discounting rate

# memory
# number of experiences stored in the memory when initialized
pretrain_length = batch_size
memory_size = 1000000  # number of experiences the memory can keep

# preprocessing hyperparameters
stack_size = 4

# turn this to true if you want to render the environment during training
episode_render = True
