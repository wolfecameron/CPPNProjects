from __future__ import print_function
import neat
import numpy as np
import os
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import spsolve
import matplotlib
matplotlib.use('TkAgg')
from matplotlib import pyplot as plt
from matplotlib import colors
from scipy.special import expit

NGEN = 0
NUMX = 0
NUMY = 0

try:
    NGEN = int(raw_input("How many generations would you like to evolve the picture?"))
except:
    NGEN = 10

try:
    tmp = raw_input("Aspect ratio of picture ('X-Y'): ")
    for i in range(len(tmp)):
        if(tmp[i] == '-'):
            NUMX = int(tmp[:i])
            NUMY = int(tmp[i + 1:])
            print(NUMX)
            print(NUMY)
            raw_input("correct?")
except:
    NUMX = 200
    NUMY = 200


xIn = []
yIn = []


# sigmoid activation to always pass activations through
def sigmoid_act(x):
    return expit(x)


# configures inputs
for y in range(0, NUMY):
    for x in range(0, NUMX):
        xIn.append(x)
        yIn.append(y)

# must normalize all inputs for CPPN to create fluid structures
tmp = np.array(xIn, copy=True)
# NOTE: mean/std for xIn and yIn will always be the same
MEAN = np.mean(tmp)
STD = np.std(tmp)


normX = []
normY = []

# creates input lists with the normalized vectors
for y in range(0, NUMY):
    for x in range(0, NUMX):
        normX.append((x - MEAN) / STD)
        normY.append((y - MEAN) / STD)


def eval_genomes(genomes, config):

    for genome_id, genome in genomes:
        genome.fitness = 0
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        outList = []
        # must unzip all the elements and run the network for each
        # input in input list
        for z in range(len(xIn)):
            # always passes output through sigmoid
            in_tuple = (normX[z], normY[z])
            x = sigmoid_act(net.activate(in_tuple)[0])
            outList.append(x)

        plt.ion()
        fig, ax = plt.subplots()
        x = np.array(outList, copy=True)
        im = ax.imshow(-x.reshape((NUMX, NUMY)).T, cmap='gray',
                       interpolation='none', norm=colors.Normalize(vmin=-1, vmax=0))
        try:
            genome.fitness += int(raw_input("rate structure 1-10: "))
        except:
            genome.fitness = 0
            print("A non-integer for fitness was entered. A value of 0 was assigned to genome fitness.")



# runs the evolver
config = neat.Config(neat.DefaultGenome, neat.DefaultReproduction,
                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                     'config-topOpt')

# Create the population, which is the top-level object for a NEAT run.
p = neat.Population(config)

# Add a stdout reporter to show progress in the terminal.
p.add_reporter(neat.StdOutReporter(True))
stats = neat.StatisticsReporter()
p.add_reporter(stats)

# Run for 100 generations.
winner = p.run(eval_genomes, n=NGEN)
