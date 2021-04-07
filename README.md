# prettyNEAT
![swingup](demo/img/swing.gif) ![biped](demo/img/biped.gif)
![ant](demo/img/ant.gif) ![racer](demo/img/race.gif)


Neuroevolution of Augmenting Topologies (NEAT) algorithm in numpy, built for multicore use and OpenAI's gym interface.

Original paper by Ken Stanley and Risto Miikkulainen: [Evolving Neural Networks Through Augmenting Topologies](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.28.5457&rep=rep1&type=pdf)

Implementation created by [Adam
Gaier](https://scholar.google.com/citations?user=GGyARB8AAAAJ&hl=en) and
originally released as part of the [Google Brain Tokyo
Workshop](https://github.com/google/brain-tokyo-workshop).

## Installation

prettyNEAT can be downloaded from pypi:

`pip install prettyNEAT`

Or installed locally

``` sh
git clone https://github.com/d9w/prettyNEAT
cd prettyNEAT
python setup.py install
```

### Other dependencies

The provided example scripts which evolve individuals for `gym` environments
have further dependencies, including `mpi4py` for distributed evaluation. To
install these additional dependencies, do:

`pip install -r requirements.txt`

## Running NEAT

![swingup](demo/img/swing.gif) ![swingup](demo/img/swing.gif)


The 'cartpole_swingup' task doesn't have any dependencies and is set as the default task, try it with the default parameters:

Training command:
```
python evolution.py
```

To view the performance of a trained controller (default `log/test_best.out` loaded):

```
python evaluation.py
```

To load and test a specific network:
```
python neat_test.py -i demo/swingup.out
```


## Data Gathering and Visualization

Data about each run is stored by default in the `log` folder with the `test` prefix, though a new prefix can be specified:

```
python evolution.py -o myExperiment_
```
Output files will still be placed in the 'log' folder but prepended with the 'myExperiment_' prefix

In addition to the best performing individual, prettyNEAT regularly updates a `_stats.out` file with run statistics. These statistics are stored as comma seperated values, and some helper functions are shown to display these statistics as well as the topology of the evolved networks.

see `prettyNeat_demo.ipynb` notebook for example usage.

## Distributed evaluation

prettyNeat uses an ask/tell pattern to handle parallelization:

```
  neat = Neat(hyp)  # Initialize Neat with hyperparameters
  for gen in range(hyp['maxGen']):        
    pop = neat.ask()            # Get newly evolved individuals from NEAT  
    reward = batchMpiEval(pop)  # Send population to workers to evaluate
    neat.tell(reward)           # Send fitness values back to NEAT    
```

The number of workers can be specified when called from the command line:

```
python evo_distributed.py -n 8
```


Algorithm hyperparameters are stored in a .json file. Default parameters specified with `-d`, modification with a `-p`:

```
python evo_distributed.py -d config/neat_default.json
```

or to use default except for certain changes

```
python evo_distributed.py -p config/swingup.json       # Swing up with standard parameters
python evo_distributed.py -p config/swing_allAct.json  # Swing up but allow hidden nodes to have several activations
```
The full list of hyperparameters is explained in [hypkey.txt](config/hypkey.txt)

---
## Extensions and Differences from Canonical NEAT

A few key differences and common extensions from the original NEAT paper have been included:

- Compatibility threshold update
    - The compatibility threshold is regularly updated to keep the number of species near a desired number. Though use of this update is widespread and mentioned on the [NEAT User's Page](https://www.cs.ucf.edu/~kstanley/neat.html), to my knowledge it has never been explicitly mentioned in a publication.

- Activation functions
    - Unless a specific activation function to be used by all hidden nodes is specified in the hyperparameters, when a new node is created it can chosen from a list of activation functions defined by the task. A probability of mutating the activation function is also defined. This allows the code to easily handle extensions for HyperNEAT and CPPN experiments.
    
- Rank-based Fitness
    - The canonical NEAT uses raw fitness values to determine the relative fitness of individuals and species. This can cause scaling problems, and can't handle negative fitness values. PrettyNEAT instead ranks the population and assigns each individual a real-valued fitness based on this ranking.

- Multi-objective
    - Many extensions of NEAT involve optimizing for additional objectives (age, number of connections, novelty, etc) and we include non-dominated sorting of the population by multiple objectives. The probability that these alternate objectives are applied can also be tuned (e.g. normal optimization, but 20% chance of ranking based on fitness _and_ number of connections). This can be used with or without speciation.
    
- Weight Tuning with CMA-ES
    - Networks produced by PrettyNEAT are exported in the form of weight matrices and a vector of activation functions. We provide an interface to further tune the weights of these networks with CMA-ES: 
    
    ```
    python cmaes.py -i log/test_best.out
    ```

--- 
## Other NEAT resources

- [Weight Agnostic Neural Networks (WANN)](https://github.com/google/brain-tokyo-workshop/tree/master/WANNRelease)
- [NEAT Software](http://eplex.cs.ucf.edu/neat_software/)
- [NEAT gym](https://github.com/simondlevy/NEAT-Gym)
- [NEAT python](https://github.com/CodeReclaimers/neat-python)
- [PyTorch NEAT](https://github.com/uber-research/PyTorch-NEAT)
