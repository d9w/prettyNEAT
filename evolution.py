import os
import sys
import time
import math
import argparse
import subprocess
import numpy as np
np.set_printoptions(precision=2, linewidth=160) 

# prettyNeat
from prettyNEAT import * # NEAT
from domain import *   # Task environments


# -- Run NEAT ------------------------------------------------------------ -- #
def master(): 
  """Main NEAT optimization script
  """

def gatherData(data,neat,gen,hyp,savePop=False):
  """Collects run data, saves it to disk, and exports pickled population

  Args:
    data       - (DataGatherer)  - collected run data
    neat       - (Neat)          - neat algorithm container
      .pop     - [Ind]           - list of individuals in population    
      .species - (Species)       - current species
    gen        - (ind)           - current generation
    hyp        - (dict)          - algorithm hyperparameters
    savePop    - (bool)          - save current population to disk?

  Return:
    data - (DataGatherer) - updated run data
  """
  data.gatherData(neat.pop, neat.species)
  if (gen%hyp['save_mod']) is 0:
    data = checkBest(data)
    data.save(gen)

  if savePop is True: # Get a sample pop to play with in notebooks    
    global fileName
    pref = 'log/' + fileName
    import pickle
    with open(pref+'_pop.obj', 'wb') as fp:
      pickle.dump(neat.pop,fp)

  return data

def checkBest(data):
  """Checks better performing individual if it performs over many trials.
  Test a new 'best' individual with many different seeds to see if it really
  outperforms the current best.

  Args:
    data - (DataGatherer) - collected run data

  Return:
    data - (DataGatherer) - collected run data with best individual updated


  * This is a bit hacky, but is only for data gathering, and not optimization
  """
  global filename, hyp
  if data.newBest is True:
    bestReps = max(hyp['bestReps'], (nWorker-1))
    rep = np.tile(data.best[-1], bestReps)
    fitVector = batchMpiEval(rep, sameSeedForEachIndividual=False)
    trueFit = np.mean(fitVector)
    if trueFit > data.best[-2].fitness:  # Actually better!      
      data.best[-1].fitness = trueFit
      data.fit_top[-1]      = trueFit
      data.bestFitVec = fitVector
    else:                                # Just lucky!
      prev = hyp['save_mod']
      data.best[-prev:]    = data.best[-prev]
      data.fit_top[-prev:] = data.fit_top[-prev]
      data.newBest = False
  return data

if __name__ == "__main__":
  ''' Parse input and launch '''
  parser = argparse.ArgumentParser(description=('Evolve NEAT networks'))
  
  parser.add_argument('-d', '--default', type=str,\
   help='default hyperparameter file', default='config/default_neat.json')

  parser.add_argument('-p', '--hyperparam', type=str,\
   help='hyperparameter file', default=None)

  parser.add_argument('-o', '--outPrefix', type=str,\
   help='file name for result output', default='test')
  
  parser.add_argument('-n', '--num_worker', type=int,\
   help='number of cores to use', default=8)

  args = parser.parse_args()
  fileName    = args.outPrefix
  hyp_default = args.default
  hyp_adjust  = args.hyperparam

  hyp = loadHyp(pFileName=hyp_default)
  updateHyp(hyp,hyp_adjust)

  data = DataGatherer(fileName, hyp)
  neat = Neat(hyp)

  for gen in range(hyp['maxGen']):
    pop = neat.ask()            # Get newly evolved individuals from NEAT
    reward = np.empty(len(pop), dtype=np.float64)
    for i in range(len(pop)):
      task = GymTask(games[hyp['task']], nReps=hyp['alg_nReps'])
      wVec   = pop[i].wMat.flatten()
      aVec   = pop[i].aVec.flatten()
      reward[i] = task.getFitness(wVec, aVec) # process it
    neat.tell(reward)           # Send fitness to NEAT

    data = gatherData(data,neat,gen,hyp)
    print(gen, '\t - \t', data.display())

  # Clean up and data gathering at run end
  data = gatherData(data,neat,gen,hyp,savePop=True)
  data.save()
  data.savePop(neat.pop,fileName) # Save population as 2D numpy arrays
