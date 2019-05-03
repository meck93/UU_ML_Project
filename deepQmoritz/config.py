# model hyperparameters
state_size = [90, 96, 4]  # 4 stacked frames
learning_rate = 0.00025  # alpha (aka learning rate)

# training hyperparameters
TOTAL_EPISODES = 50  # total episodes for training
MAX_STEPS = 5000  # max possible steps in an episode
STUCK_STEPS = 1500  # max steps in the same x_position until Mario is considered stuck
batch_size = 16

# exploration parameters
explore_start = 1.0  # exploration probability at start
explore_stop = 0.01  # minimum exploration probability
decay_rate = 0.00001  # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.9  # discounting rate

# memory
# number of experiences stored in the memory when initialized
pretrain_length = batch_size
memory_size = 1000000  # number of experiences the memory can keep

# preprocessing hyperparameters
stack_size = 4

# turn this to true if you want to render the environment during training
episode_render = True
