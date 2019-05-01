import retro  # pip install gym-retro
import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
import neat  # pip install neat-python
import pickle  # pip install cloudpickle

import gym_super_mario_bros
#from gym_super_mario_bros.actions import SIMPLE_MOVEMENT #(`RIGHT_ONLY`, `SIMPLE_MOVEMENT`, and `COMPLEX_MOVEMENT`)
#/Users/andrastefanianegus/miniconda3/envs/supermario/lib/python3.7/site-packages/gym_super_mario_bros-7.1.6.dist-info
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

#based on : sample XOR w/ NEAT code in the python-neat module: https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/evolve-feedforward-parallel.py
# and Sonic the Hedgehog tutorial code fr. https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT/blob/master/neat-paralle-sonic.py

simple_movement= [[None],['LEFT'], ['RIGHT'], ['RIGHT', 'A'],
                     ['A'],['B'],['UP','RIGHT'],['UP']] #,['LEFT','UP'],['UP'],['UP',None],['RIGHT','UP'],['RIGHT', 'B'], ['RIGHT', 'A', 'B'], ['LEFT'],

#[['NOOP'], ['right'], ['right', 'A'], ['right', 'B'], ['right', 'A', 'B'], ['A'], ['left']]

#A parallel version using neat.parallel.
class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):

        #self.env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
        self.env = retro.make(game='SuperMarioBros3-Nes', state="1Player.World1.Level1.state")  # , state='Level1-1', record=True)
        #self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env, simple_movement)

        self.env.reset()

        ob, _, _, _ = self.env.step(self.env.action_space.sample())

        inx = int(ob.shape[0] / 8)
        iny = int(ob.shape[1] / 8)
        done = False

        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)

        fitness = 0
        xpos = 0
        xpos_max = 0
        counter = 0
        imgarray = []
        #self.env.render()
        coins_init=0
        stage_init = 1
        status_init ='small'


        score_max=0


        while not done:
            #self.env.render()
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            imgarray = np.ndarray.flatten(ob)

            actions = net.activate(imgarray)
            #print("++++++++++++++++++++++++++++++++++++++++++++++++")
            #print("HERE HERE HERE HERE HERE", self.env.step(self.env.action_space.sample()))
            #print("++++++++++++++++++++++++++++++++++++++++++++++++")

            ob, rew, done, info= self.env.step(self.env.action_space.sample())
            #ob, rew, done, info = self.env.step(actions)
            #print('rew =', rew)
            '''
            xpos = info['x_pos'] # info['x']
            coins= info['coins']
            status =info['status'] #{'small', 'tall', 'fireball'}
            stage= info['stage'] #The current stage, i.e., _{1, ..., 4}_
            '''
            score = info['score']


            #if xpos > xpos_max:
            if score> score_max:
                score_max = score
                counter = 0
                fitness += 1
            else:
                counter += 1

            if counter > 250:
                done = True

            '''if xpos >= info['screen_x_end']:
                fitness += 100000
                done = True
            '''

        #print("fitness =",fitness)
        return fitness


def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward2')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

#Either create new population(command above), or restore from checkpoint
#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-93')

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

# Run for up to 100 generations.
pe = neat.ParallelEvaluator(5, eval_genomes) #neat.ParallelEvaluator(10, eval_genomes)
winner = p.run(pe.evaluate, 22)

# Display the winning genome.
print('\nBest genome:\n{!s}'.format(winner))


with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

