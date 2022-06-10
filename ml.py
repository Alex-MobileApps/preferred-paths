import numpy as np
from brain import Brain, GlobalBrain
from preferred_path import PreferredPath
import torch
from torch import tensor as T
from torch.nn import Sequential, Linear, ReLU, Module
from torch.distributions import Normal
from utils import device
from datetime import datetime
from typing import List

class BrainDataset():
    def __init__(self, sc: np.ndarray, fc: np.ndarray, euc_dist: np.ndarray, hubs: np.ndarray, regions: np.ndarray, func_regions: np.ndarray, fns: List[str], fn_weights: List[int] = None):
        """
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
            brain = GlobalBrain(sc[i], fc[i], euc_dist, hubs=hubs, regions=regions, func_regions=func_regions)
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
            self.adj[i] = T(brain.sc_bin[triu_i].astype(int), dtype=torch.int).to(device)
            self.sp[i] = T(brain.shortest_paths(), dtype=torch.float).to(device)
            self.pp[i] = PreferredPath(adj=brain.sc_bin, fn_vector=fn_vector, fn_weights=weights)
            self.sample_idx[i] = np.column_stack(np.where(brain.fc_bin > 0))


    @staticmethod
    def fn_mapper(name, brain):
        if name == 'streamlines':
            vals = brain.streamlines()
            return lambda loc, nxt, prev_nodes, target: vals[loc,nxt]
        if name == 'node_str':
            vals = brain.node_strength(weighted=True)
            return lambda loc, nxt, prev_nodes, target: vals[nxt]
        if name == 'target_node':
            vals = brain.is_target_node
            return lambda loc, nxt, prev_nodes, target: vals(nxt, target)
        if name == 'target_region':
            vals = brain.is_target_region
            return lambda loc, nxt, prev_nodes, target: vals(nxt, target)
        if name == 'hub':
            vals = brain.hubs(binary=True)
            return lambda loc, nxt, prev_nodes, target: vals[nxt]
        if name == 'neighbour_just_visited_node':
            vals = brain.neighbour_just_visited_node
            return lambda loc, nxt, prev_nodes, target: vals(nxt, prev_nodes)
        if name == 'edge_con_diff_region':
            vals = brain.edge_con_diff_region
            return lambda loc, nxt, prev_nodes, target: vals(loc, nxt, target)
        if name == 'inter_regional_connections':
            vals = brain.inter_regional_connections(weighted=False, distinct=True)
            return lambda loc, nxt, prev_nodes, target: vals[nxt]
        if name == 'prev_visited_region':
            vals = brain.prev_visited_region
            return lambda loc, nxt, prev_nodes, target: vals(loc, nxt, prev_nodes)
        if name == 'target_func_region':
            vals = brain.is_target_func_region
            return lambda loc, nxt, prev_nodes, target: vals(nxt, target)
        if name == 'edge_con_diff_func_region':
            vals = brain.edge_con_diff_func_region
            return lambda loc, nxt, prev_nodes, target: vals(loc, nxt, target)
        if name == 'prev_visited_func_region':
            vals = brain.prev_visited_func_region
            return lambda loc, nxt, prev_nodes, target: vals(loc, nxt, prev_nodes)
        if name == 'inter_func_regional_connections':
            vals = brain.inter_func_regional_connections(weighted=False, distinct=True)
            return lambda loc, nxt, prev_nodes, target: vals[nxt]
        if name == 'rand_walk':
            return lambda loc, nxt, prev_nodes, target: 1
        if name == 'closest_to_target':
            vals = brain.closest_to_target
            return lambda loc, nxt, prev_nodes, target: vals(loc, nxt, target)
        raise ValueError(f'{name} is an invalid function')

    def __len__(self):
        return len(self.adj)

    def __getitem__(self, idx):
        return (self.adj[idx], self.sp[idx], self.pp[idx], self.sample_idx[idx])


class PolicyEstimator(Module):
    def __init__(self, res, fn_len, hidden_units=20, init_weight=None):
        """
        Parameters
        ----------
        res : int
            Brain resolution (i.e. number of nodes)
        fn_len : int
            Number of function criteria
        hidden_units : int, optional
            Number of units in the hidden layer, by default 20
        init_weight : int, optional
            Initial weight for all edges in the neural network, by default None
            If None, weights are set randomly
        """

        super(PolicyEstimator, self).__init__()
        self.n_inputs = int(res * (res - 1) / 2)
        self.n_outputs = fn_len * 2 # includes both mean and sigma
        self.network = Sequential(
            Linear(self.n_inputs, hidden_units),
            ReLU(),
            Linear(hidden_units, self.n_outputs))

        # Set a fixed network weight
        if init_weight is not None:
            for layer in [0,2]:
                torch.nn.init.constant_(self.network[layer].weight, init_weight)

    def predict(self, state):
        return self.network(state)


def global_reward(pred, sp):
    """
    Returns a reward, defined by the global navigation efficiency ratio

    Parameters
    ----------
    pred : numpy.ndarray
        Predicted path lengths vector
    sp : numpy.ndarray
        Shortest path lengths vector

    Returns
    -------
    float
        Global navigation efficiency ratio
    """

    # Local navigation efficiency ratio
    local = torch.zeros(sp.size, dtype=torch.float).to(device)
    mask = torch.where(pred != -1).to(device)
    local[mask] = sp[mask] / pred[mask]

    # Global navigation efficiency ratio
    return local.sum() / sp.size


def local_reward(pred, sp):
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


def reinforce(pe, opt, data, epochs, batch, lr, sample=0, plt_data=None, save_path=None, save_freq=1, log=False, path_method=PreferredPath._DEF_METHOD):
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
        epoch_fn(pe, opt, data, batch, sample, num_fns, plt_data, log, path_method)

        # Save
        if save_path:
            if e % save_freq == 0:
                save(save_path, pe, opt, plt_data)

        if log:
            print('\rDone')


def epoch_fn(pe, opt, data, batch, sample, num_fns, plt_data, log, path_method):
    t1 = datetime.now() # Track epoch duration
    offset = 0
    while offset + batch <= len(data):
        rewards = torch.zeros((batch,1), dtype=torch.float).to(device)
        success = torch.zeros(batch, dtype=torch.float).to(device)
        adj, sp, pp, sample_idx = data[offset:offset+batch]

        # Find criteria mu and sig
        probs = pe.predict(adj)
        mu, sig = probs[:,:num_fns], abs(probs[:,num_fns:]) + 1

        # Sample a set of criteria weights
        N = Normal(mu, sig)
        actions = N.sample().to(device)

        # Batch
        for i in range(batch):
            if log:
                print(f'\r{str(i+1+offset)}', end='')
            pp[i].fn_weights = actions[i].tolist()
            rewards[i], success[i] = sample_batch_fn(pp[i], sp[i], sample, sample_idx[i], path_method) if sample > 0 else full_batch_fn(pp[i], sp[i], path_method)

        # Step
        step_fn(opt, N, actions, rewards)

        # Update plotting data
        if plt_data is not None:
            plt_data['rewards'].append(rewards.mean().item())
            plt_data['success'].append(success.mean())

            # Rescale to range -1 <= mu <= 1 for results
            max_mu = abs(mu).max(axis=1)[0].reshape(-1,1)
            scaled_mu = mu / max_mu
            scaled_sig = sig / max_mu # Var(aX) = a^2 Var(X)
            for j in range(num_fns):
                plt_data['mu'][j].append(scaled_mu[:,j].mean().item())
                plt_data['sig'][j].append(scaled_sig[:,j].mean().item())

        # Run next batch
        offset += batch

    # Track epoch data
    t2 = datetime.now()
    plt_data['epoch_seconds'].append((t2-t1).seconds)
    plt_data['epochs'] += 1


def sample_batch_fn(pp, sp, sample, sample_idx, path_method):
    rewards = torch.zeros(sample, dtype=torch.float).to(device)
    success = torch.zeros(sample, dtype=torch.float).to(device)
    len_sample_idx = len(sample_idx)
    path_method = pp._convert_method_to_fn(path_method)

    for i in range(sample):
        sp_val = 0
        while sp_val <= 0:
            s, t = sample_idx[np.random.choice(len_sample_idx)]
            sp_val = sp[s,t]
        pred = PreferredPath._single_path_formatted(path_method, s, t, False)
        rewards[i] = local_reward(pred, sp_val)
        success[i] = pred != -1

    return rewards.mean(), success.mean()


def full_batch_fn(pp, sp, path_method):
    mask = torch.where(sp > 0).to(device)
    pred = pp.retrieve_all_paths(method=path_method)[mask]
    rewards = global_reward(pred, sp[mask])
    success = 1 - (pred == -1).sum() / len(pred)
    return rewards, success


def step_fn(opt, N, actions, rewards):
    opt.zero_grad()
    loss = -N.log_prob(actions) * (rewards - 0.5)
    loss = loss.mean()
    loss.backward()
    opt.step()


def save(path, pe, opt, plt_data):
    save_data = {
        'model_state_dict': pe.network.state_dict(),
        'optimizer_state_dict': opt.state_dict()}
    for key, value in plt_data.items():
        save_data[key] = value
    torch.save(save_data, path)