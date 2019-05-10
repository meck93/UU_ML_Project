# current session
MODEL_NAME = "Prio3"
RECORDING_NAME = './recordings/SuperMarioBros3-Nes-1Player.World1.Level1-000008.bk2'

# model hyperparameters
HEIGHT = 96
WIDTH = 96
N_FRAMES = 4
state_size = [HEIGHT, WIDTH, N_FRAMES]  # 4 stacked frames

# training hyperparameters
learning_rate = 2.5e-4  # alpha (aka learning rate)
TOTAL_EPISODES = 10  # total episodes for training
MAX_STEPS = 1000  # max possible steps in an episode
STUCK_STEPS = 250  # max steps in the same x_position until Mario is considered stuck
TOTAL_TIMESTEPS = TOTAL_EPISODES * MAX_STEPS
batch_size = 32

# exploration parameters
explore_start = 1.0  # exploration probability at start
explore_stop = 0.02  # minimum exploration probability
decay_rate = 0.00001  # exponential decay rate for exploration prob

# Q learning hyperparameters
gamma = 0.95  # discounting rate

# reward parameters
# mostly in OpenAI's scenario.json file
PENALTY_DYING = -15.0  # custom dying penalty

# memory
# number of experiences stored in the memory when initialized
pretrain_length = batch_size * 32
memory_size = 50000  # total number of experiences the memory can keep

# prioritized experience replay parameters
REPLAY_ALPHA = 0.6
REPLAY_BETA0 = 0.4
REPLAY_BETA0_ITERS = TOTAL_EPISODES * (MAX_STEPS + STUCK_STEPS) // 2
REPLAY_EPS = 1e-6

# preprocessing hyperparameters
stack_size = 4

# turn this to true if you want to render the environment during training
episode_render = True
