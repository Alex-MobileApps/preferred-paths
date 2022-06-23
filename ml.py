import numpy as np
from brain import Brain
from preferred_path import PreferredPath
import torch
from torch import tensor as T
from torch.nn import Sequential, Linear, ReLU, Module
from torch.distributions import Normal
from utils import device
from datetime import datetime
from typing import List, Tuple

class BrainDataset():
    def __init__(self, sc: np.ndarray, fc: np.ndarray, euc_dist: np.ndarray, hubs: np.ndarray, regions: np.ndarray, func_regions: np.ndarray, fns: List[str], fn_weights: List[int] = None):
        """
        Creates a new BrainDataset object

        Parameters
        ----------
        sc, fc : np.ndarray
            Connectivity matrices for each subject
            3D with shape: (number of subjects, resolution, resolution)
        euc_dist : np.ndarray
            Euclidean distance matrix
            2D with shape: (resolution, resolution)
        hubs : np.ndarray
            Array with the indexes of the hub nodes
        regions : np.ndarray
            Array with the region number each node is assigned to
        func_regions : np.ndarray
            Array with the functional region number each node is assigned to
        fns : List[str]
            List of strings with the function name to use for ML. Valid functions include:
            'streamlines', 'node_str', 'target_node', 'target_region', 'hub', 'neighbour_just_visited_node', 'edge_con_diff_region', 'inter_regional_connections', 'prev_visited_region', 'target_func_region', 'edge_con_diff_func_region', 'prev_visited_func_region'
            And their 'anti' versions, e.g. 'anti_streamlines', 'anti_node_str', etc.
        fn_weights : List[int], optional
            Initial criteria function weights, by default None
            If None, random weights will be generated instead
        """

        n = len(sc)
        res = len(euc_dist)
        num_fns = len(fns)

        # Init vars
        triu = int(res * (res - 1) / 2)
        triu_i = np.triu_indices(res, 1)
        self.adj = torch.zeros((n, triu), dtype=torch.float).to(device)
        self.sp = torch.zeros((n, res, res), dtype=torch.float).to(device)
        self.pp = [None] * n
        self.sample_idx = [None] * n

        # Fill vars
        for i in range(n):
            brain = Brain(sc[i], fc[i], euc_dist, hubs=hubs, regions=regions, func_regions=func_regions)
            fn_vector = [None] * num_fns
            for j, name in enumerate(fns):
                fn_vector[j] = BrainDataset.fn_mapper(name, brain)

            # Initialise weights from -1 to 1
            if fn_weights is None:
                weights = list(np.random.random(size=num_fns) * 2 - 1) # Change scale from 0..1 to -1..1
            else:
                weights = fn_weights
            max_weight = abs(max(weights, key=abs))
            weights = [w / max_weight for w in weights] # Set largest to -1 or 1

            # Store vars
            tmp_sp = brain.shortest_paths()
            self.adj[i] = T(brain.sc_bin[triu_i].astype(int), dtype=torch.int).to(device)
            self.pp[i] = PreferredPath(adj=brain.sc_bin, fn_vector=fn_vector, fn_weights=weights)
            self.sp[i] = T(tmp_sp, dtype=torch.float).to(device)

            # FC edge locations where there is a connected SC path
            self.sample_idx[i] = np.column_stack(np.where((brain.fc_bin > 0) & (tmp_sp > 0)))


    @staticmethod
    def fn_mapper(name: str, brain: 'Brain') -> 'function':
        """
        Extract a criteria function for a brain

        Parameters
        ----------
        name : str
            Criteria function name
            Available names: 'streamlines', 'node_str', 'target_node', 'target_region', 'hub', 'neighbour_just_visited_node', 'edge_con_diff_region', 'inter_regional_connections', 'prev_visited_region', 'target_func_region', 'edge_con_diff_func_region', 'prev_visited_func_region', 'inter_func_regional_connections', 'rand_walk', 'closest_to_target'
            And their 'anti' versions, e.g. 'anti_streamlines', 'anti_node_str', etc.
        brain : Brain
            Brain to get the criteria function for

        Returns
        -------
        function
            Function to calculate the criteria function for a potential next node from the current location, potential next node, previous nodes and target node

        Raises
        ------
        ValueError
            If criteria name is invalid
        """

        if name.startswith('anti_'):
            name = name[5:]
            mult = -1 # Penalise non-anti version
        else:
            mult = 1 # Reward non-anti version
        if name == 'streamlines':
            vals = brain.streamlines()
            return lambda loc, nxt, prev_nodes, target: mult * vals[loc,nxt]
        if name == 'node_str':
            vals = brain.node_strength(weighted=True)
            return lambda loc, nxt, prev_nodes, target: mult * vals[nxt]
        if name == 'target_node':
            vals = brain.is_target_node
            return lambda loc, nxt, prev_nodes, target: mult * vals(nxt, target)
        if name == 'target_region':
            vals = brain.is_target_region
            return lambda loc, nxt, prev_nodes, target: mult * vals(nxt, target)
        if name == 'hub':
            vals = brain.hubs(binary=True)
            return lambda loc, nxt, prev_nodes, target: mult * vals[nxt]
        if name == 'neighbour_just_visited_node':
            vals = brain.neighbour_just_visited_node
            return lambda loc, nxt, prev_nodes, target: mult * vals(nxt, prev_nodes)
        if name == 'edge_con_diff_region':
            vals = brain.edge_con_diff_region
            return lambda loc, nxt, prev_nodes, target: mult * vals(loc, nxt, target)
        if name == 'inter_regional_connections':
            vals = brain.inter_regional_connections(weighted=False, distinct=True)
            return lambda loc, nxt, prev_nodes, target: mult * vals[nxt]
        if name == 'prev_visited_region':
            vals = brain.prev_visited_region
            return lambda loc, nxt, prev_nodes, target: mult * vals(loc, nxt, prev_nodes)
        if name == 'target_func_region':
            vals = brain.is_target_func_region
            return lambda loc, nxt, prev_nodes, target: mult * vals(nxt, target)
        if name == 'edge_con_diff_func_region':
            vals = brain.edge_con_diff_func_region
            return lambda loc, nxt, prev_nodes, target: mult * vals(loc, nxt, target)
        if name == 'prev_visited_func_region':
            vals = brain.prev_visited_func_region
            return lambda loc, nxt, prev_nodes, target: mult * vals(loc, nxt, prev_nodes)
        if name == 'inter_func_regional_connections':
            vals = brain.inter_func_regional_connections(weighted=False, distinct=True)
            return lambda loc, nxt, prev_nodes, target: mult * vals[nxt]
        if name == 'rand_walk':
            return lambda loc, nxt, prev_nodes, target: mult * 1
        if name == 'closest_to_target':
            vals = brain.closest_to_target
            return lambda loc, nxt, prev_nodes, target: mult * vals(loc, nxt, target)
        raise ValueError(f'{name} is an invalid function')


    def __len__(self) -> int:
        """
        Number of brains in the dataset

        Returns
        -------
        int
            Number of brains in the dataset
        """

        return len(self.adj)


    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, list, list]:
        """
        Returns data for a brain at a specified index in the dataset
        Includes: unweighted adjacency matrix, shortest path matrix, preferred-path model, FC edge indexes

        Parameters
        ----------
        idx : int
            Index of the brain in the dataset

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, list, list]
            Adjacency matrix, shortest path matrix, preferred-path model, FC edge indexes
        """

        return (self.adj[idx], self.sp[idx], self.pp[idx], self.sample_idx[idx])


class PolicyEstimator(Module):
    def __init__(self, res: int, fn_len: int, hidden_units: int = 20, init_weight: float = None, const_sig: float = None):
        """
        Creates a neural network for a continuous action space policy gradient

        Parameters
        ----------
        res : int
            Brain resolution (i.e. number of nodes)
        fn_len : int
            Number of function criteria
        hidden_units : int, optional
            Number of units in the hidden layer, by default 20
        init_weight : float, optional
            Initial weight for all edges in the neural network, by default None
            If None, weights are set randomly
        const_sig : float, optional
            Set a fixed standard deviation sigma (pre-scaling) for each criteria, by default None
            Affects the number of output units in the network and whether or not the standard deviation is learnt (None) or kept constant (not None)
        """

        super(PolicyEstimator, self).__init__()
        self.n_inputs = int(res * (res - 1) / 2)
        self.n_outputs = fn_len
        if const_sig is None:
            self.n_outputs *= 2 # includes both mean and sigma

        # Create neural network
        self.network = Sequential(
            Linear(self.n_inputs, hidden_units),
            ReLU(),
            Linear(hidden_units, self.n_outputs))

        # Set a fixed network weight
        if init_weight is not None:
            for layer in [0,2]:
                torch.nn.init.constant_(self.network[layer].weight, init_weight)


    def predict(self, state: torch.Tensor) -> torch.Tensor:
        """
        Returns the forward pass output from the neural network

        Parameters
        ----------
        state : torch.Tensor
            Sequence of unweighted adjacency matrices for each brain in the batch

        Returns
        -------
        torch.Tensor
            Forward pass output
        """

        return self.network(state)


def global_reward(pred: np.ndarray, sp: np.ndarray) -> float:
    """
    Returns a reward, defined by the global navigation efficiency ratio

    Parameters
    ----------
    pred : np.ndarray
        Predicted path lengths vector
    sp : np.ndarray
        Shortest path lengths vector

    Returns
    -------
    float
        Global navigation efficiency ratio
    """

    # Local navigation efficiency ratio
    len_sp = len(sp)
    local = torch.zeros(len_sp, dtype=torch.float).to(device)
    mask = np.where(pred != -1)
    local[mask] = sp[mask] / pred[mask]

    # Global navigation efficiency ratio
    return local.sum() / len_sp


def local_reward(pred: int, sp: int) -> float:
    """
    Returns a reward, defined by the local navigation efficiency ratio

    Parameters
    ----------
    pred : int
        Predicted path length
    sp : int
        Shortest path length

    Returns
    -------
    float
        Local navigation efficiency ratio
    """

    if pred > 0:
        return sp / pred
    return 0


def reinforce(pe: 'PolicyEstimator', opt: torch.optim, data: 'BrainDataset', epochs: int, batch: int, lr: float, sample: int = 0, const_sig: float = None, pos_only: bool = False, plt_data: dict = None, save_path: str = None, save_freq: int = 1, log: bool = False, path_method: str = PreferredPath._DEF_METHOD) -> None:
    """
    Runs the continuous policy gradient reinforce algorithm

    Parameters
    ----------
    pe : PolicyEstimator
        Policy estimator
    opt : torch.optim
        Neural network optimiser
    data : BrainDataset
        Training dataset
    epochs : int
        Number of passes to run on each brain
    batch : int
        Number of brains to include in each batch
    lr : float
        Learning rate
    sample : int
        Number of path samples to take per brain (0 to use full brain)
    const_sig : float, optional
        Set a fixed standard deviation sigma (pre-scaling) for each criteria, by default None
        If None, sigma will be learnt during training
    pos_only : bool, optional
        Whether or not to take the absolute value of criteria weights, by default False
        Only use when mixing regular and anti criteria
        - If a criteria is known to produce a positive weight (e.g. target node), the anti criteria isn't required
        - If a criteria may produce a negative weight, the anti criteria is required (the negative weighted version will then plateau at 0 instead of overlapping)
    plt_data : dict
        Dictionary to store data as the network progresses.
        Requires the keys: 'rewards', 'success', 'mu' and 'sig' with values being a list.
        The values for 'mu' and 'sig' should be a 2D list, with a row for each criteria in the model
    save_path : str
        Location to save the state of the neural network and optimiser as well as plt_data
    save_freq : int
        Number of epochs to complete before each save operation (only used when save_path is set)
    log : bool
        Whether or not to continuously print the current epoch and batch
    path_method : str
        'rev'  : Revisits allowed. If a revisit occurs, that the path sequence equals 'None' due to entering an infinite loop
        'fwd'  : Forward only, nodes cannot be revisited and backtracking isn't allowed
        'back' : Backtracking allowed, nodes cannot be revisited and backtracking to previous nodes occur at dead ends to find alternate routes
    """

    # Setup
    num_fns = data.pp[0].fn_length

    # Update learning rate
    for g in opt.param_groups:
        g['lr'] = lr

    # Epoch
    for _ in range(epochs):
        e = plt_data['epochs'] + 1
        if log:
            print(f'\r-- Epoch {e} --')
        epoch_fn(pe=pe, opt=opt, data=data, batch=batch, sample=sample, const_sig=const_sig, num_fns=num_fns, pos_only=pos_only, plt_data=plt_data, log=log, path_method=path_method)

        # Save
        if save_path:
            if e % save_freq == 0:
                save(path=save_path, pe=pe, opt=opt, plt_data=plt_data)

        if log:
            print('\rDone')


def epoch_fn(pe: 'PolicyEstimator', opt: torch.optim, data: 'BrainDataset', batch: int, sample: int, const_sig: float, num_fns: int, pos_only: bool, plt_data: dict, log: bool, path_method: str):
    """
    Performs an epoch of training

    Parameters
    ----------
    num_fns : int
        Number of criteria functions
    See 'reinforce' for more information on parameters
    """

    t1 = datetime.now() # Track epoch duration
    offset = 0
    if const_sig is not None:
        sig = torch.ones((1,num_fns)) * const_sig
    while offset + batch <= len(data):
        rewards = torch.zeros((batch,1), dtype=torch.float).to(device)
        success = torch.zeros(batch, dtype=torch.float).to(device)
        adj, sp, pp, sample_idx = data[offset:offset+batch]

        # Action
        probs = pe.predict(adj)

        # Extract mu
        mu = probs[:,:num_fns]
        if pos_only:
            mu = abs(mu)

        # Extract sigma
        if const_sig is None:
            sig = abs(probs[:,num_fns:]) + 1

        # Sample a set of criteria weights
        N = Normal(mu, sig)
        actions = N.sample().to(device)

        # Batch
        for i in range(batch):
            if log:
                print(f'\r{str(i+1+offset)}', end='')
            pp[i].fn_weights = actions[i].tolist()
            rewards[i], success[i] = sample_batch_fn(pp=pp[i], sp=sp[i], sample=sample, sample_idx=sample_idx[i], path_method=path_method) if sample > 0 else full_batch_fn(pp=pp[i], sp=sp[i], sample_idx=sample_idx[i], path_method=path_method)

        # Step
        step_fn(opt=opt, N=N, actions=actions, rewards=rewards)

        # Update plotting data
        if plt_data is not None:
            plt_data['rewards'].append(rewards.mean().item())
            plt_data['success'].append(success.mean())
            for j in range(num_fns):
                plt_data['mu'][j].append(mu[:,j].mean().item())
                plt_data['sig'][j].append(sig[:,j].mean().item())

        # Run next batch
        offset += batch

    # Track epoch data
    t2 = datetime.now()
    plt_data['epoch_seconds'].append((t2-t1).seconds)
    plt_data['epochs'] += 1


def sample_batch_fn(pp: 'PreferredPath', sp: torch.Tensor, sample: int, sample_idx: np.ndarray, path_method: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the average reward and success from a number of FC edges sampled randomly with replacement in a Brain

    Parameters
    ----------
    pp : PreferredPath
        Preferred path algorithm for the brain
    sp : torch.Tensor
        Shortest paths matrix for the brain
    sample : int
        Number of FC edge samples
    sample_idx : np.ndarray
        Indexes of all FC edge source and target pairs in the brain
    path_method : str
        See 'reinforce' for more information

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple containing the average reward and success as single element tensors
    """

    rewards = torch.zeros(sample, dtype=torch.float).to(device)
    success = torch.zeros(sample, dtype=torch.float).to(device)
    len_sample_idx = len(sample_idx)
    path_method = pp._convert_method_to_fn(path_method)

    # Predict random FC samples
    for i in range(sample):

        # Source and target node
        s, t = sample_idx[np.random.choice(len_sample_idx)]

        # Compute rewards and success
        pred = PreferredPath._single_path_formatted(path_method, s, t, False)
        rewards[i] = local_reward(pred=pred, sp=sp[s,t])
        success[i] = pred != -1

    # Compute average reward and success
    return rewards.mean(), success.mean()


def full_batch_fn(pp: 'PreferredPath', sp: torch.Tensor, sample_idx: np.ndarray, path_method: str) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the average reward and success from all FC edges in a Brain

    Parameters
    ----------
    pp : PreferredPath
        Preferred path algorithm for the brain
    sp : torch.Tensor
        Shortest paths matrix for the brain
    sample_idx : np.ndarray
        Indexes of all FC edge source and target pairs in the brain
    path_method : str
        See 'reinforce' for more information

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Tuple containing the average reward and success as single element tensors
    """

    path_method = pp._convert_method_to_fn(path_method)

    # Predict all FC edges
    pred = torch.zeros(len(sample_idx), dtype=torch.float).to(device)
    for i, (s,t) in enumerate(sample_idx):
        pred[i] = PreferredPath._single_path_formatted(path_method, s, t, False)

    # Compute average reward and success
    rewards = global_reward(pred=pred, sp=sp[sample_idx[:,0],sample_idx[:,1]])
    success = 1 - (pred == -1).sum() / len(pred)
    return rewards, success


def step_fn(opt: torch.optim, N: torch.distributions.Normal, actions: torch.Tensor, rewards: torch.Tensor) -> None:
    """
    Performs a backpropagation step to update the neural network weights

    Parameters
    ----------
    opt : torch.optim
        Neural network optimiser
    N : torch.distributions.Normal
        Normal distribution for the criteria
    actions : torch.Tensor
        Samples selected from the normal distribution
    rewards : torch.Tensor
        Rewards for each action
    """

    opt.zero_grad()
    loss = -N.log_prob(actions) * (rewards - 0.5)
    loss = loss.mean()
    loss.backward()
    opt.step()


def save(path: str, pe: 'PolicyEstimator', opt: torch.optim, plt_data: dict):
    """
    Saves the current state of training to file

    Parameters
    ----------
    path : str
        Where to save the current state
    See 'reinforce' for more information on parameters
    """

    save_data = {
        'model_state_dict': pe.network.state_dict(),
        'optimizer_state_dict': opt.state_dict()}
    for key, value in plt_data.items():
        save_data[key] = value
    torch.save(save_data, path)