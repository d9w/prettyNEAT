import numpy as np
import math
import copy
import json

from .utils import *
from .nsga_sort import nsga_sort
from .ind import Ind


class Neat():
  """NEAT main class. Evolves population given fitness values of individuals.
  """
  def __init__(self, hyp):
    """Intialize NEAT algorithm with hyperparameters
    Args:
      hyp - (dict) - algorithm hyperparameters

    Attributes:
      p       - (dict)     - algorithm hyperparameters (see config/hypkey.txt)
      pop     - (Ind)      - Current population
      species - (Species)  - Current species   
      innov   - (np_array) - innovation record
                [5 X nUniqueGenes]
                [0,:] == Innovation Number
                [1,:] == Source
                [2,:] == Destination
                [3,:] == New Node?
                [4,:] == Generation evolved
      gen     - (int)      - Current generation
    """
    self.p       = hyp
    self.pop     = [] 
    self.species = [] 
    self.innov   = [] 
    self.gen     = 0  

  ''' Subfunctions '''
  from ._variation import evolvePop, recombine
  from ._speciate  import Species, speciate, compatDist,\
                          assignSpecies, assignOffspring  

  def ask(self):
    """Returns newly evolved population
    """
    if len(self.pop) == 0:
      self.initPop()      # Initialize population
    else:
      self.probMoo()      # Rank population according to objectivess
      self.speciate()     # Divide population into species
      self.evolvePop()    # Create child population 

    return self.pop       # Send child population for evaluation

  def tell(self,reward):
    """Assigns fitness to current population

    Args:
      reward - (np_array) - fitness value of each individual
               [nInd X 1]

    """
    for i in range(np.shape(reward)[0]):
      self.pop[i].fitness = reward[i]
      self.pop[i].nConn   = self.pop[i].nConn
  
  def initPop(self):
    """Initialize population with a list of random individuals
    """
    ##  Create base individual
    p = self.p # readability
    
    # - Create Nodes -
    nodeId = np.arange(0,p['ann_nInput']+ p['ann_nOutput']+1,1)
    node = np.empty((3,len(nodeId)))
    node[0,:] = nodeId
    
    # Node types: [1:input, 2:hidden, 3:bias, 4:output]
    node[1,0]             = 4 # Bias
    node[1,1:p['ann_nInput']+1] = 1 # Input Nodes
    node[1,(p['ann_nInput']+1):\
           (p['ann_nInput']+p['ann_nOutput']+1)]  = 2 # Output Nodes
    
    # Node Activations
    node[2,:] = p['ann_initAct']
    # - Create Conns -
    nConn = (p['ann_nInput']+1) * p['ann_nOutput']
    ins   = np.arange(0,p['ann_nInput']+1,1)            # Input and Bias Ids
    outs  = (p['ann_nInput']+1) + np.arange(0,p['ann_nOutput']) # Output Ids
    
    conn = np.empty((5,nConn,))
    conn[0,:] = np.arange(0,nConn,1)      # Connection Id
    conn[1,:] = np.tile(ins, len(outs))   # Source Nodes
    conn[2,:] = np.repeat(outs,len(ins) ) # Destination Nodes
    conn[3,:] = np.nan                    # Weight Values
    conn[4,:] = 1                         # Enabled?
        
    # Create population of individuals with varied weights
    pop = []
    for i in range(p['popSize']):
        newInd = Ind(conn, node)
        newInd.conn[3,:] = (2*(np.random.rand(1,nConn)-0.5))*p['ann_absWCap']
        newInd.conn[4,:] = np.random.rand(1,nConn) < p['prob_initEnable']
        newInd.express()
        newInd.birth = 0
        pop.append(copy.deepcopy(newInd))  
    # - Create Innovation Record -
    innov = np.zeros([5,nConn])
    innov[0:3,:] = pop[0].conn[0:3,:]
    innov[3,:] = -1
    
    self.pop = pop
    self.innov = innov

  def probMoo(self):
    """Rank population according to Pareto dominance.
    """
    # Compile objectives
    meanFit = np.asarray([ind.fitness for ind in self.pop])
    nConns  = np.asarray([ind.nConn   for ind in self.pop])
    nConns[nConns==0] = 1 # No connections is pareto optimal but boring...
    objVals = np.c_[meanFit,1/nConns] # Maximize

    # Alternate between two objectives and single objective
    if self.p['alg_probMoo'] < np.random.rand():
      rank = nsga_sort(objVals[:,[0,1]])
    else: # Single objective
      rank = rankArray(-objVals[:,0])

    # Assign ranks
    for i in range(len(self.pop)):
      self.pop[i].rank = rank[i]
