from .task_gym import GymTask
from .config import games
import json
import numpy as np

def loadHyp(pFileName, printHyp=False):
  """Loads hyperparameters from .json file
  Args:
      pFileName - (string) - file name of hyperparameter file
      printHyp  - (bool)   - print contents of hyperparameter file to terminal?

  Note: see config/hypkey.txt for detailed hyperparameter description
  """
  with open(pFileName) as data_file: hyp = json.load(data_file)

  # Task hyper parameters
  task = GymTask(games[hyp['task']],paramOnly=True)
  hyp['ann_nInput']   = task.nInput
  hyp['ann_nOutput']  = task.nOutput
  hyp['ann_initAct']  = task.activations[0]
  hyp['ann_absWCap']  = task.absWCap
  hyp['ann_mutSigma'] = task.absWCap * 0.2
  hyp['ann_layers']   = task.layers # if fixed toplogy is used

  if hyp['alg_act'] == 0:
    hyp['ann_actRange'] = task.actRange
  else:
    hyp['ann_actRange'] = np.full_like(task.actRange,hyp['alg_act'])



  if printHyp is True:
    print(json.dumps(hyp, indent=4, sort_keys=True))
  return hyp

def updateHyp(hyp,pFileName=None):
  """Overwrites default hyperparameters with those from second .json file
  """
  if pFileName != None:
    print('\t*** Running with hyperparameters: ', pFileName, '\t***')
    with open(pFileName) as data_file: update = json.load(data_file)
    hyp.update(update)

    # Task hyper parameters
    task = GymTask(games[hyp['task']],paramOnly=True)
    hyp['ann_nInput']   = task.nInput
    hyp['ann_nOutput']  = task.nOutput
    hyp['ann_initAct']  = task.activations[0]
    hyp['ann_absWCap']  = task.absWCap
    hyp['ann_mutSigma'] = task.absWCap * 0.1
    hyp['ann_layers']   = task.layers # if fixed toplogy is used


    if hyp['alg_act'] == 0:
      hyp['ann_actRange'] = task.actRange
    else:
      hyp['ann_actRange'] = np.full_like(task.actRange,hyp['alg_act'])
