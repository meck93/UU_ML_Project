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
        self.genome = genome
        self.config = config



    def work(self):
        
        self.env = retro.make(game='SuperMarioBros3-Nes', state="1Player.World1.Level1.state") 
        
        self.env.reset()
        
        ob, _, _, _ = self.env.step(self.env.action_space.sample())
        
        inx = int(ob.shape[0]/8)
        iny = int(ob.shape[1]/8)
        done = False
        
        net = neat.nn.FeedForwardNetwork.create(self.genome, self.config)#neat.nn.recurrent.RecurrentNetwork did not work
        
        fitness = 0
        xpos = 0
        xpos_max = 0
        counter = 0
        imgarray = []
        score_max =0
        x_multi_max=0
        x_multi =0

        while not done:
            #self.env.render() #uncomment this to see Mario move
            #print( 'posit: ' , ram[0x6d] * 0x100 + ram[0x86])
            ob = cv2.resize(ob, (inx, iny))
            ob = cv2.cvtColor(ob, cv2.COLOR_BGR2GRAY)
            ob = np.reshape(ob, (inx, iny))
            
            imgarray = np.ndarray.flatten(ob)
            
            actions = net.activate(imgarray)
            
            ob, rew, done, info = self.env.step(actions)
            
            #xpos goes fr. 0-255, and resets to 0;
            #uses 'xpos_multiplier' to keep track of how many times xpos reset to 0 b/c it moved to right
            xpos = info['xpos']+255*info['xpos_multiplier']
            score=info['score']
            
            if xpos > xpos_max:
                #print('xpos:', xpos,"\n")
                xpos_max = xpos
                counter = 0
                fitness += 1

            elif score>score_max:
                score_max =score
                counter =0
                fitness +=0.1 

            else:
                counter += 1
                
            if counter > 250 or info['lives']==-1:
                done = True
                
            '''if xpos >= 2600: #an approximation of the screen_x_end for level 1
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

p = neat.Population(config)

#Either create new population(command above), or restore from checkpoint
#p = neat.Checkpointer.restore_checkpoint('neat-checkpoint-30')

p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)
p.add_reporter(neat.Checkpointer(10))

# evaluating 5 in parallel; change this if you want 
pe = neat.ParallelEvaluator(5, eval_genomes)

#run for 65 generations
winner = p.run(pe.evaluate,65)
localtime = time.localtime(time.time())
winnerfile ='winner'+str(localtime.tm_mon)+str(localtime.tm_mday)+'_'+str(localtime.tm_hour)+str(localtime.tm_min)+'.pkl'
with open(winnerfile, 'wb') as output:
    pickle.dump(winner, output, 1)

