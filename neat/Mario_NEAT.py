import retro        # pip install gym-retro
import numpy as np  # pip install numpy
import cv2          # pip install opencv-python
import neat         # pip install neat-python
import pickle       # pip install cloudpickle

import time

#make sure you add the data.json into the correct folder. For example, mine is in /miniconda3/envs/supermario/lib/python3.7/site-packages/retro/data/stable/SuperMarioBros3-Nes


# based on : sample XOR w/ NEAT code in the python-neat module: https://github.com/CodeReclaimers/neat-python/blob/master/examples/xor/evolve-feedforward-parallel.py
# and Sonic the Hedgehog tutorial code fr. https://gitlab.com/lucasrthompson/Sonic-Bot-In-OpenAI-and-NEAT/blob/master/neat-paralle-sonic.py
# and #https://medium.freecodecamp.org/how-to-use-ai-to-play-sonic-the-hedgehog-its-neat-9d862a2aef98

class Worker(object):
    def __init__(self, genome, config):
        #self.genome = genome
        self.genome = pickle.load(open('winner59_1818BEST.pkl', 'rb')) #pickle.load(open('winner516_1755.pkl', 'rb'))##genome ##pickle.load(open('winner58_2058.pkl', 'rb')) #genome #pickle.load(open('winner430_1958noscore.pkl', 'rb')) #genome #
        self.config = config


    def work(self):
        
        self.env = retro.make(game='SuperMarioBros3-Nes', state="1Player.World1.Level1.state")
        
        self.env.reset()
        
        ob, _, _, _ = self.env.step(self.env.action_space.sample())
        
        inx = int(ob.shape[0]/8)
        iny = int(ob.shape[1]/8)
        done = False

        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)#neat.nn.recurrent.RecurrentNetwork did sounot work
        
        fitness = 0
        #xpos = 0
        xpos_max = 0
        counter = 0
        #inputs = []
        score_max =0
        #x_multi_max=0
        #x_multi =0
        lives_max=4

        while not done:
            self.env.render() #uncomment this to see Mario move

            #scaled=cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            #scaled =cv2.resize(scaled,(inx,iny))
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))

            #cv2.imshow('main',scaled)
            #cv2.waitKey(1)


            #copy of the array collapsed into one dimension
            inputs = np.ndarray.flatten(ob)

            #inputs=ob.get_scaled_state()
            #action = net.activate(inputs)
            #Feeds inputs into the network and returns resulting outputs or actions
            actions = net.activate(inputs)
            
            ob, rew, done, info = self.env.step(actions)

            lives_current = info['lives']
            #xpos goes fr. 0-255, and resets to 0;
            #uses 'xpos_multiplier' to keep track of how many times xpos reset to 0 b/c it moved further to right
            if info['xpos_multiplier'] < 32:
                xpos = info['xpos']+255*info['xpos_multiplier']
                #print('xpos ',xpos, '\n')
            else:
                xpos = info['xpos']

            score = info['score']

            if xpos > xpos_max:
                #print('xpos:', xpos,"\n")
                xpos_max = xpos
                counter = 0
                fitness += 1

            elif score > score_max:
                score_max = score
                counter = 0
                fitness += 0.1
                #print('fitness', fitness)

            elif lives_current<lives_max:
                lives_max=lives_current
                counter=0
                fitness-=15

            else:
                counter += 1

            #character death happens at lives==-1
            if counter > 250 or info['lives'] == -1: #
                #print( 'counter=',counter,'\n')
                done = True
                
            '''if xpos >= 2500: #an approximation of the screen_x_end for level 1
                fitness += 100000
                done = True
            '''
                
        #print(fitness)
        return fitness


def eval_genomes(genome, config):
    run_eval = Worker(genome, config)
    return run_eval.work()


config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction, 
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-feedforward')

#Either create new population, or restore from checkpoint
p = neat.Population(config)
#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-95')

# stats and saving checkpoints
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
#p.add_reporter(neat.Checkpointer(generation_interval=8, time_interval_seconds=1200))

# Runs evaluation functions in parallel subprocesses in order to
# evaluate multiple genomes at once.
# Currently evaluating 6 in p6, eval_genomes)

pe = neat.ParallelEvaluator(1, eval_genomes)
#pe = neat.ParallelEvaluator(1, eval_genomes)

#run for 100 (n) generations, if n=None it runs until solution/ extinction
#run(fitness_function, n=None)
# evaluate = Distributes the evaluation jobs among the subprocesses,
# then assigns each fitness back to the appropriate genome.
best_genome = p.run(pe.evaluate, 25)
#best_genome = p.run(pe.evaluate, 1)
#pe.stop()
#pe.stop()

#Save winner
localtime = time.localtime(time.time())
winnerfile ='winner'+str(localtime.tm_mon)+str(localtime.tm_mday)+'_'+str(localtime.tm_hour)+str(localtime.tm_min)+'.pkl'
with open(winnerfile, 'wb') as output:
    pickle.dump(best_genome, output, 1)