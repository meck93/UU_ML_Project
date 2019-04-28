import retro  # pip install gym-retro
import numpy as np  # pip install numpy
import cv2  # pip install opencv-python
import neat  # pip install neat-python
import pickle  # pip install cloudpickle

import gym_super_mario_bros
from gym_super_mario_bros.actions import SIMPLE_MOVEMENT
#/Users/andrastefanianegus/miniconda3/envs/supermario/lib/python3.7/site-packages/gym_super_mario_bros-7.1.6.dist-info
from nes_py.wrappers import BinarySpaceToDiscreteSpaceEnv

#based on code fr. https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT/blob/master/neat-paralle-sonic.py

class Worker(object):
    def __init__(self, genome, config):
        self.genome = genome
        self.config = config

    def work(self):

        #self.env = retro.make('SonicTheHedgehog-Genesis', 'GreenHillZone.Act1')
        #self.env = retro.make(game='SuperMarioBros3-Nes', state="1Player.World1.Level1.state")  # , state='Level1-1', record=True)
        self.env = gym_super_mario_bros.make('SuperMarioBros-v0')
        self.env = BinarySpaceToDiscreteSpaceEnv(self.env, SIMPLE_MOVEMENT)

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
            print('rew =', rew)
            xpos = info['x_pos'] # info['x']
            coins= info['coins']
            status =info['status'] #{'small', 'tall', 'fireball'}
            stage= info['stage'] #The current stage, i.e., _{1, ..., 4}_

            if xpos > xpos_max:
                xpos_max = xpos
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

        print("fitness =",fitness)
        return fitness


def eval_genomes(genome, config):
    worky = Worker(genome, config)
    return worky.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward2')

p = neat.Population(config)

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

pe = neat.ParallelEvaluator(10, eval_genomes)

winner = p.run(pe.evaluate)

with open('winner.pkl', 'wb') as output:
    pickle.dump(winner, output, 1)

