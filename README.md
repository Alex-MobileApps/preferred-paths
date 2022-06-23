
# Preferred paths <!-- omit in toc -->

Used to compute and evaluate navigation paths in the human brain from a weighted set of navigation criteria.

- [Setup](#setup)
  - [Requirements](#requirements)
  - [Downloading](#downloading)
- [Brain Objects](#brain-objects)
  - [Brain Constructor](#brain-constructor)
  - [Brain Criteria](#brain-criteria)
- [PreferredPath Objects](#preferredpath-objects)
  - [Path Navigation Methods](#path-navigation-methods)
  - [PreferredPath Constructor](#preferredpath-constructor)
  - [Finding a single path](#finding-a-single-path)
  - [Finding all paths](#finding-all-paths)
- [Machine Learning](#machine-learning)
  - [BrainDataset Objects](#braindataset-objects)
  - [PolicyEstimator Objects](#policyestimator-objects)
  - [Running REINFORCE](#running-reinforce)
  - [Saving Results](#saving-results)
  - [Loading Results](#loading-results)
  - [Visualising Results](#visualising-results)
  - [Full Example](#full-example)
- [OzStar Training](#ozstar-training)
  - [OzStar Setup](#ozstar-setup)
  - [OzStar Scripts](#ozstar-scripts)
  - [OzStar Running](#ozstar-running)
  - [Ozstar Downloading Results](#ozstar-downloading-results)

## Setup

### Requirements

- Python 3
- NumPy
- SciPy
- PyTorch
- Matplotlib

### Downloading

- Install all requirements listed above
- Clone the contents of https://github.com/alex-mobileapps/preferred-paths
- Create a new python script (e.g. demo.py) in the 'preferred-paths' directory to work from

```
git clone https://github.com/alex-mobileapps/preferred-paths
cd preferred-paths
touch demo.py
```

## Brain Objects

Brain objects are used to extract features from a brain, such as node strength and streamlines.
This requires the SC, FC and Euclidean distance matrices for that brain, as well as vectors with the hub nodes and each node's region and functional regions.

### Brain Constructor

> Brain(sc: np.ndarray, fc: np.ndarray, euc_dist: np.ndarray, sc_directed: bool = False, sc_thresh: float = 1.0, fc_thresh: float = 0.01, sc_thresh_type: str = 'pos', fc_thresh_type: str = 'pos', hubs: np.ndarray = None, regions: np.ndarray = None, func_regions: np.ndarray = None)

```
import numpy as np
from brain import Brain

sc = np.array(
    [[ 0, 5, 0, 0, 0, 0, 0, 0],
     [ 0, 0, 4, 0, 0, 9, 0, 0],
     [ 0, 4, 0, 0, 0, 1, 7, 0],
     [ 0, 0, 0, 0, 2, 6, 7, 0],
     [ 0, 0, 0, 0, 0, 3, 0, 0],
     [ 0, 9, 1, 6, 3, 0, 0, 0],
     [ 0, 0, 7, 7, 0, 0, 0, 0],
     [ 0, 0, 0, 0, 0, 0, 0, 0]])

fc = np.array(
    [[ 0,  0,  1,  0, -2,  0,  2,  0],
     [ 0,  0, -2, -1,  0,  0, -4,  0],
     [ 1, -2,  0, -2,  0, -1, 17,  2],
     [ 0, -1, -2,  0,  1, -1,  1, -2],
     [-2,  0,  0,  1,  0,  6, -1,  2],
     [ 0,  0, -1, -1,  6,  0,  5, 13],
     [ 2, -4, 17,  1, -1,  5,  0, 11],
     [ 0,  0,  2, -2,  2, 13, 11,  0]])

euc_dist = np.array(
    [[ 0, 31, 63, 34, 29, 39, 76, 32],
     [31,  0, 60, 30, 26, 38, 83, 45],
     [63, 60,  0, 64, 62, 67, 96, 67],
     [34, 30, 64,  0, 29, 38, 69, 46],
     [29, 26, 62, 29,  0, 33, 82, 40],
     [39, 38, 67, 38, 33,  0, 77, 46],
     [76, 83, 96, 69, 82, 77,  0, 76],
     [32, 45, 67, 46, 40, 46, 76,  0]])

hubs = [1, 4, 5]

regions = [0, 0, 1, 2, 2, 2, 3, 4]

func_regions = [0, 0, 0, 1, 1, 2, 2, 2]

brain = Brain(sc=sc, fc=fc, euc_dist=euc_dist, sc_directed=True, sc_thresh=1, fc_thresh=0.01, hubs=hubs, regions=regions, func_regions=func_regions)
```

<img src="img/sc_wei.png">

### Brain Criteria

More information on parameters and additional criteria can found in docstrings in the code

#### Streamlines <!-- omit in toc -->

> Brain.streamlines(weighted: bool = True) -> np.ndarray

Number of streamlines between any two nodes
```
streamlines = brain.streamlines(weighted=True)
result = streamlines[0,1] # streamlines from node 0 to node 1

print(result)
# 5
```

#### Node Strength <!-- omit in toc -->

> Brain.node_strength(weighted: bool = True, method: str = 'tot') -> np.ndarray

Number of streamlines attached to each node
```
node_str = brain.node_strength(weighted=True, method='tot')
result = node_str[3] # node strength of node 3 for both in and out degree

print(result)
# 28
```

#### Hubs <!-- omit in toc -->

> Brain.hubs(binary: bool = False) -> np.ndarray

Hub nodes in the brain
```
result1 = brain.hubs(binary=False)  # List of hub nodes
result2 = brain.hubs(binary=True)   # Whether or not each node is a hub node

print(result1)
# [1 4 5]

print(result2)
# [0 1 0 0 1 1 0 0]
```

#### Target Node <!-- omit in toc -->

> Brain.is_target_node(nxt: int, target: int) -> int

Whether or not the potential next node is the target node
```
result = brain.is_target_node(nxt=0, target=2) # Whether or not node 0 is the target node 2

print(result)
# 0
```

#### Neighbour of the Just Visited Node <!-- omit in toc -->

> Brain.neighbour_just_visited_node(nxt: int, prev_nodes: List[int]) -> int

Returns whether or not a potential next node neighbours the most recently visited node
```
result1 = brain.neighbour_just_visited_node(nxt=1, prev_nodes=[4,5])     # Whether or not node 1 neighbours the previously visited node
result2 = brain.neighbour_just_visited_node(nxt=6, prev_nodes=[4,5])     # Whether or not node 6 neighbours the previously visited node

print(result1)
# 1

print(result2)
# 0
```

#### Target Region <!-- omit in toc -->

> Brain.is_target_region(nxt: int, target: int) -> int

Returns whether or not a potential next node is in the target node's region
```
result = brain.is_target_region(nxt=3, target=5) # Whether or not node 3 is in the same region as target node 5

print(result)
# 1
```

#### Edges Connecting Different Regions <!-- omit in toc -->

> Brain.edge_con_diff_region(loc: int, nxt: int, target: int) -> int

Returns whether or not a potential next node leaves the current region, if it is not already in the target region
```
result = brain.edge_con_diff_region(loc=0, nxt=2, target=3)    # Whether or not node 2 leaves a non-target region

print(result)
# 1
```

#### Inter Regional Connections <!-- omit in toc -->

> Brain.inter_regional_connections(weighted: bool = True, distinct: bool = False) -> np.ndarray

Returns how many connections each node has to different regions
```
irc = brain.inter_regional_connections(weighted=True)
result = irc[1] # Number of weighted streamlines node 1 has to different regions

print(result)
# 13
```

#### Previously Visited Region <!-- omit in toc -->

> Brain.prev_visited_region(loc: int, nxt: int, prev_nodes: List[int]) -> int

Returns whether or not the region of a potential next node has already been visited, unless it remains in the same region
```
result = brain.prev_visited_region(loc=3, nxt=4, prev_nodes=[2,6]) # Whether or node 4 is from a previously visited non-target region

print(result)
# 0
```

#### Target Functional Region <!-- omit in toc -->

> Brain.is_target_func_region(nxt: int, target: int) -> int

Returns whether or not a potential next node is in the target node's functional region
```
result = brain.is_target_func_region(nxt=5, target=7) # Whether or not node 5 in the functional region of target node 7

print(result)
# 1
```

#### Edges Connecting Different Functional Regions <!-- omit in toc -->

> Brain.edge_con_diff_func_region(loc: int, nxt: int, target: int) -> int

Returns whether or not a potential next node leaves the current functional region, if it is not already in the target functional region
```
result = brain.edge_con_diff_func_region(loc=0, nxt=3, target=5) # Whether or not node 3 leaves a non-target functional region

print(result)
# 1
```

#### Previously Visited Functional Regions <!-- omit in toc -->

> Brain.prev_visited_func_region(loc: int, nxt: int, prev_nodes: List[int]) -> int

Returns whether or not the functional region of a potential next node has already been visited, unless it remains in the same functional region
```
result = brain.prev_visited_func_region(loc=1, nxt=2, prev_nodes=[3,5]) # Whether or not node 2 is in a previously visited non-target functional region

print(result)
# 0
```

#### Inter Functional Region Connections <!-- omit in toc -->

> Brain.inter_func_regional_connections(weighted: bool = True, distinct: bool = False) -> np.ndarray

Returns how many connections each node has to different functional regions
```
ifrc = brain.inter_func_regional_connections(weighted=True)
result = ifrc[5] # Number of connections to different functional regions

print(result)
# 19
```

#### Closest to Target <!-- omit in toc -->

> Brain.closest_to_target(self, loc: int, nxt: int, target: int) -> int

Returns the change in Euclidean distance to the target node when moving to the next node
```
result = brain.closest_to_target(loc=0, nxt=1, target=2) # How much closer node 2 becomes after moving to node 1

print(result)
# 3
```

#### Shortest Paths <!-- omit in toc -->

> Brain.shortest_paths(method: str = 'hops') -> np.ndarray

Lengths of the shortest paths between nodes
```
sp = brain.shortest_paths(method='hops')
result = sp[0,2] # Fewest number of hops from node 0 to 2

print(result)
# 2
```

## PreferredPath Objects

PreferredPath objects are used to compute paths in a brain through the guidance of a weighted set of brain criteria.

When determining paths, each brain criteria must be a function that accepts the following parameters:
1. Location of the current node - int
2. Location of the next node being assessed - int
3. Previously visited node locations (excluding current node) - List[int]
4. Target node location - int

The output of each criteria function is assessed and normalised for each potential next node.
These outputs are then weighted and summed to create a 'score' the preference of each node as the next node in the path.
The node with the highest score is visited next.

### Path Navigation Methods

- **rev**: Nodes can be revisited in a path (paths fail on revisit cycles, dead-ends and disconnected graphs)
- **fwd**: Nodes cannot be revisited in a path (paths fail on dead-ends and disconnected graphs)
- **back**: Same as 'fwd', except backing out of dead-ends are allowed to find alternative routes (paths fail on disconnected graphs)

### PreferredPath Constructor

Construction requires a brain's unweighted adjacency matrix, criteria functions and the weighting for each criteria function.

> PreferredPath(adj: np.ndarray, fn_vector: List['function'], fn_weights: List[float], validate: bool = True)

```
from preferred_path import PreferredPath

# Criteria functions (using streamlines and node strength)
node_str = brain.node_strength(weighted=False)
streamlines = brain.streamlines()
fn_vector = [
    lambda loc, nxt, prev_nodes, target: streamlines[loc,nxt],
    lambda loc, nxt, prev_nodes, target: node_str[nxt]]

# Criteria weights
fn_weights = [0.4, 0.7]

# Preferred path object
pp = PreferredPath(adj=brain.sc_bin, fn_vector=fn_vector, fn_weights=fn_weights)
```

### Finding a single path

> PreferredPath.retrieve_single_path(source: int, target: int, method: str = 'fwd', out_path: bool = False) -> List[int] or int

```
result1 = pp.retrieve_single_path(source=0, target=5, method='rev', out_path=False) # Length of path between nodes 0 and 5
result2 = pp.retrieve_single_path(source=0, target=5, method='rev', out_path=True)  # Sequence of nodes in the path between nodes 0 and 5

print(result1)
# 2

print(result2)
# [0, 1, 5]
```

### Finding all paths

> PreferredPath.retrieve_all_paths(method: str = 'fwd', out_path: bool = False) -> dict or np.ndarray

```
result1 = pp.retrieve_all_paths(method='back', out_path=False)   # Length of paths between any 2 nodes
result2 = pp.retrieve_all_paths(method='back', out_path=True)    # Sequence of nodes in any path

print(result1)
# [[ 0.  1.  5.  3.  4.  2.  4. -1.]
#  [-1.  0.  4.  2.  3.  1.  3. -1.]
#  [-1.  2.  0.  2.  3.  1.  3. -1.]
#  [-1.  2.  3.  0.  2.  1.  4. -1.]
#  [-1.  2.  3.  5.  0.  1.  4. -1.]
#  [-1.  1.  2.  4.  5.  0.  3. -1.]
#  [-1.  2.  1.  4.  5.  3.  0. -1.]
#  [-1. -1. -1. -1. -1. -1. -1.  0.]]

print(result2)
# {
#     0: {
#         1: [0, 1],                # Sequence of nodes in the path between nodes 0 and 1
#         2: [0, 1, 5, 3, 6, 2]
#         ...
#     }
#     ...
# }
```

## Machine Learning

### BrainDataset Objects

BrainDataset objects hold the Brain objects as well as methods that are used in training.

> BrainDataset(sc: np.ndarray, fc: np.ndarray, euc_dist: np.ndarray, hubs: np.ndarray, regions: np.ndarray, func_regions: np.ndarray, fns: List[str], fn_weights: List[int] = None)

- **sc**, **fc**: 3D matrices ($m \times n \times n$), containing the sc and fc matrices for all brains being trained
- **fns**: List of brain criteria functions to use. Available functions include:
  - streamlines
  - node_str
  - hub
  - target_node
  - neighbour_just_visited_node
  - target_region
  - edge_con_diff_region
  - inter_regional_connections
  - prev_visited_region
  - target_func_region
  - edge_con_diff_func_region
  - prev_visited_func_region
  - inter_func_regional_connections
  - rand_walk (i.e. selecting any neighbour with equal probability)
  - closest_to_target
  - anti criteria versions of any of the above (e.g. anti_streamlines, anti_node_str)
- All other parameters are the same as for PreferredPath objects

```
from ml import BrainDataset

sc_3d = sc.reshape((1,*sc.shape))
fc_3d = fc.reshape((1,*sc.shape))
fns = ['streamlines','node_str','target_node']

dataset = BrainDataset(sc=sc_3d, fc=fc_3d, euc_dist=euc_dist, hubs=hubs, regions=regions, func_regions=func_regions, fns=fns)
```

### PolicyEstimator Objects

PolicyEstimator objects contain the neural network

> PolicyEstimator(res: int, fn_len: int, hidden_units: int = 10, init_weight: float = None, const_sig: float = None)

```
from ml import PolicyEstimator

pe = PolicyEstimator(res=len(sc), fn_len=len(fns), hidden_units=10)
```

### Running REINFORCE

Learn a policy that optimses the weighting for each of the brain criteria functions.

> reinforce(pe: 'PolicyEstimator', opt: torch.optim, data: 'BrainDataset', epochs: int, batch: int, lr: float, sample: int = 0, const_sig: float = None, pos_only: bool = False, plt_data: dict = None, save_path: str = None, save_freq: int = 1, log: bool = False, path_method: str = 'fwd') -> None

```
import torch
from ml import reinforce

# Holds training evolution results
plt_data = {
  'epochs': 0,
  'epoch_seconds': [],
  'rewards': [],
  'success': [],
  'mu': [[] for _ in range(len(fns))],
  'sig': [[] for _ in range(len(fns))],
  'fns': fns}

# Gradient descent optimizer
opt = torch.optim.Adam(pe.network.parameters())

# Improve weights over 1000 epochs
reinforce(pe, opt, dataset, epochs=1000, batch=1, sample=100, lr=0.001, plt_data=plt_data)
```

### Saving Results

Results can be saved as a '.pt' file during training if 'save_freq' and 'save_path' are set when running REINFORCE.
- **save_freq**: Number of epochs to complete before each save
- **save_path**: Where to save the results

```
reinforce(pe, opt, dataset, epochs=1000, batch=1, sample=100, lr=0.001, plt_data=plt_data, save_path='demo.pt', save_freq=100)
```

### Loading Results

Results can be loaded from a '.pt' file after training

```
import torch
from utils import device # detects whether you are using CPU or GPU

plt_data = torch.load('demo.pt', map_location=device)
```

The loaded data is in a dictionary with keys:
- **model_state_dict**: Neural network state (so that training can be resumed)
- **optimizer_state_dict**: Optimizer state (so that training can be resumed)
- **epochs**: Number of epochs completed
- **epoch_seconds**: Number of seconds taken to complete each epoch
- **rewards**: Observed navigation efficiency after each batch
- **success**: Observed success ratio after each batch
- **mu**: Brain criteria function weights after each batch
- **sig**: Standard deviation of the exploration space for each brain criteria function after each batch
- **fns**: List of brain criteria functions used in training (in the same order as they appear in mu and sigma)
- Any other parameters that you manually added to plt_data during training

To resume training, the state of the policy estimator and optimizer must be set
```
pe.network.load_state_dict(plt_data.pop('model_state_dict'))
opt.load_state_dict(plt_data.pop('optimizer_state_dict'))
```

### Visualising Results

#### Rewards Evolution <!-- omit in toc -->

> plot_rewards(ax: matplotlib.axes.Axes, plt_data: dict, plt_avg: int = None, plt_off: int = 0, plt_subtitle: str = '', loc: str = 'lower right', **kwargs) -> None

```
import matplotlib.pyplot as plt
from cust_plot import plot_rewards

fig, ax = plt.subplots()
plot_rewards(ax=ax, plt_data=plt_data, plt_avg=100)
```

<img src="img/reward.png">

#### Success Ratio Evolution <!-- omit in toc -->

> plot_success(ax: matplotlib.axes.Axes, plt_data: dict, plt_avg: int = None, plt_off: int = 0, plt_subtitle: str = '', loc: str = 'lower right', **kwargs) -> None

```
from cust_plot import plot_success

fig, ax = plt.subplots()
plot_success(ax=ax, plt_data=plt_data, plt_avg=100)
```

<img src="img/success.png">

#### Mu Evolution <!-- omit in toc -->

> plot_mu(ax: matplotlib.axes.Axes, plt_data: dict, plt_off: int = 0, plt_subtitle: str = '', loc: str = 'lower left', scaled: bool = True, zero_line: bool = False, **kwargs) -> None

```
from cust_plot import plot_mu

fig, ax = plt.subplots()
plot_mu(ax=ax, plt_data=plt_data)
```

<img src="img/mu.png">

#### Sigma Evolution <!-- omit in toc -->

> plot_sig(ax: matplotlib.axes.Axes, plt_data: dict, plt_off: int = 0, plt_subtitle: str = '', loc: str = 'upper right', scaled: bool = True, **kwargs) -> None

```
from cust_plot import plot_sig

fig, ax = plt.subplots()
plot_sig(ax=ax, plt_data=plt_data)
```

<img src="img/sig.png">

#### Final Mu and Sigma <!-- omit in toc -->

> plot_pdf(ax: matplotlib.axes.Axes, plt_data: dict, plt_subtitle: str = '', loc: str = 'center right', scaled: bool = True, zero_line: bool = False, **kwargs) -> None

```
from cust_plot import plot_pdf

fig, ax = plt.subplots()
plot_pdf(ax=ax, plt_data=plt_data)
```

<img src="img/pdf.png">

#### Full Results for Multiple Experiments <!-- omit in toc -->

> plot_multi_experiments(paths: List[str], plt_title: str = None, plt_avg: int = None, plt_off: int = 0, loc: str = None, scaled: bool = True, zero_line: bool = False, figsize: Tuple[int,int] = None, save_path: str = None) -> None

```
from cust_plot import plot_multi_experiments

# After creating results for demo2.pt and demo3.pt
plot_multi_experiments(paths=['demo.pt','demo2.pt','demo3.pt'], plt_avg=100, plt_title='Final Results')
```

<img src="img/multi_experiment.png">

#### Summary for Multiple Experiments <!-- omit in toc -->

> plot_summary(ax: matplotlib.axes.Axes, paths: List[str], scaled: bool = True, alpha: float = 0.7, zero_line: bool = False, plt_subtitle: str = '') -> None

```
from cust_plot import plot_summary

# After creating results for demo2.pt and demo3.pt
fig, ax = plt.subplots()
plot_summary(ax=ax, paths=['demo.pt','demo2.pt','demo3.pt'])
```

<img src="img/summary.png">

#### Summary for Multiple Experiments on Multiple Brains <!-- omit in toc -->

> plot_multi_summary(paths: List[List[str]], scaled: bool = True, alpha: float = 0.7, zero_line: bool = False, plt_title: str = None, plt_subtitle: List[str] = None, figsize: Tuple[int,int] = None, save_path: str = None)

```
from cust_plot import plot_multi_summary

# After creating results for demo2.pt, demo3.pt and demo4.pt
plot_multi_summary(paths=[['demo.pt','demo2.pt'],['demo3.pt','demo4.pt']], plt_subtitle=[' (brain 1)',' (brain 2)'])
```

<img src="img/multi_summary.png">

### Full Example

See [demo_ml.ipynb](https://github.com/Alex-MobileApps/preferred-paths/blob/main/demo_ml.ipynb)

## OzStar Training

### OzStar Setup

Copy the 'preferred-paths' directory to your OzStar home directory from the command line

```
scp -r path/to/preferred-paths your-username@ozstar.swin.edu.au:
```

### OzStar Scripts

Job scripts need to be created to run training on OzStar.
These scripts can run the ozstar_train.py file in the preferred-paths directory.

ozstar_train.py takes the following parameters
- **res**: Brain resolution size - int, by default 219
- **subj**: Brain number to train (e.g. 4 will train s004, use 0 to train all brains) - int, by default 1
- **epoch**: Number of epochs to run - int, by default 1
- **batch**: Number of brains per batch (set to one if training on 1 brain only) - int, by default 1
- **sample**: Number of FC edge samples to consider per look at a brain (set to 0 to use all FC edges instead of sampling) - int, by default 100
- **hu**: Number of hidden units in the single hidden layer of the neural network - int, by default 10
- **lr**: Learning rate - float, by default 0.001
- **save**: Path to save the results to on OzStar (e.g. demo.pt will save to your-username/preferred-path/demo.pt on OzStar) - str, by default None
- **load**: Path to load a previous set of results from to resume training (same format as 'save') - str, by default None
- **savefreq**: Number of epochs to complete between each save update - int, by default 1
- **nolog**: Flags whether to not print log output during training
- **pathmethod**: Path navigation method to use in the PreferredPath algorithm (accepts fwd, rev or back) - str, by default 'fwd'
- **seed**: Set a random seed to make the results reproducible - int, by default None
- **nnweight**: Set a fixed initial weight for all edges in the neural network - float, by default None
- **constsig**: Set a fixed standard deviation sigma instead of learning it during training - float, by default None
- **posonly**: Flags whether to prevent criteria from being weighted less than 0 (consider using anti-criteria in combination with this for known negatively weighted criteria)
- **fns**: Sequence of criteria functions to use in training. Available functions include: streamlines, node_str, hub, target_node, neighbour_just_visited_node, target_region, edge_con_diff_region, inter_regional_connections, prev_visited_region, target_func_region, edge_con_diff_func_region, prev_visited_func_region, inter_func_regional_connections and their anti-criteria versions (e.g. anti_streamlines, anti_hub, etc.)

For example:
```
#!/bin/bash
#SBATCH --job-name=demo_job
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=48:00:00
#SBATCH --mem-per-cpu=6000
#SBATCH --partition=skylake

module load python/3.8.5
module load pytorch/1.7.1-python-3.8.5
module load scipy/1.6.0-python-3.8.5
module load scikit-learn/0.24.2-python-3.8.5

python3 ~/preferred-paths/ozstar_train.py --res 219 --subj 1 --epoch 1000 --sample 100 --hu 10 --lr 0.001 --pathmethod fwd --nolog --savefreq 100 --save ~/preferred-paths/demo.pt --fns anti_streamlines target_node
```
Will create a job named demo_job that will run on the 219 resolution version of brain s001 with criteria functions anti_streamlines and target_node.
This will terminate after 48 hours or 1000 epochs (whichever comes first) and use 100 FC edge samples per step, 10 hidden units, learning rate of 0.001, fwd path navigation method, does not log ouput during each epoch, and saves to preferred-paths/demo.pt on your OzStar directory every 100 epochs. All other parameters in ozstar_train will have their default values.

You can change the job name from demo_job to easily identify it in OzStar during training as well as change the running time from 48 hours if you need more/less time.

### OzStar Running

1. Create a OzStar script as above and save it in a file (e.g. demo.sh)
2. Copy the script to the preferred-paths folder in your OzStar directory from the command line
```
scp path/to/demo.sh your-username@ozstar.swin.edu.au:preferred-paths
```
3. Login to OzStar from the command line
```
ssh your-username@ozstar.swin.edu.au
```
4. Navigate to the preferred-paths directory
```
cd preferred-paths
```
5. Run the script (will give you a job ID e.g. 27724619)
```
sbatch demo.sh
```
6. View progress of your job (will disappear once job has completed or was unsuccessful)
```
squeue -u your-username
```
7. View job output if training is unsuccessful. E.g. if job ID was 27724619
```
more slurm-27724619.out
```

### Ozstar Downloading Results

If the 'save' parameter is set in the OzStar script, two files will be saved on OzStar.
E.g. if 'save' was set to 'demo.pt':
- preferred-paths/demo.pt : Contains the training results
- preferred-paths/params_demo.pt.txt : Contains the parameters used to produce the results demo.pt

Download the results locally
```
cd path/to/save/results
scp your-username@ozstar.swin.edu.au:preferred-paths/demo.pt .
scp your-username@ozstar.swin.edu.au:preferred-paths/params_demo.pt.txt .
```