# current session
MODEL_NAME = "V2"

# model hyperparameters
HEIGHT = 32  # 76
WIDTH = 32  # 96
N_FRAMES = 4
state_size = [HEIGHT, WIDTH, N_FRAMES]  # 4 stacked frames

# training hyperparameters
learning_rate = 2.5e-4  # alpha (aka learning rate)
TOTAL_EPISODES = 1500  # total episodes for training
MAX_STEPS = 8000  # max possible steps in an episode
STUCK_STEPS = 1000  # max steps in the same x_position until Mario is considered stuck
TOTAL_TIMESTEPS = TOTAL_EPISODES * MAX_STEPS
batch_size = 32

# exploration parameters
EXPLORE_START = 1.0  # exploration probability at start
EXPLORE_STOP = 0.1  # minimum exploration probability
DECAY_RATE = 0.00001  # exponential decay rate for exploration prob
DECAY_STEPS = 1000000

# Q learning hyperparameters
GAMMA = 0.99  # discounting rate
MAX_TAU = 10000  # Tau is the C step where we update our target network

# reward parameters
# mostly in OpenAI's scenario.json file
PENALTY_DYING = -25.0  # custom dying penalty
TIME_DECAY_PENALTY = 50

# memory
# number of experiences stored in the memory when initialized
pretrain_length = batch_size * 2048
memory_size = 1000000  # total number of experiences the memory can keep

# prioritized experience replay parameters
REPLAY_ALPHA = 0.6
REPLAY_BETA0 = 0.4
REPLAY_BETA0_ITERS = 1000000
REPLAY_EPS = 1e-6

# turn this to true if you want to render the environment during training
episode_render = True
