import numpy as np
from brain import Brain
from preferred_path import PreferredPath
#import matplotlib.pyplot as plt
import torch
from torch import tensor as T
from torch.nn import Sequential, Linear, ReLU, Module
from torch.distributions import Normal
#from IPython import display
#from cust_plot import plot
from utils import device
from datetime import datetime

class BrainDataset():
    def __init__(self, sc, fc, euc_dist, hubs, regions):
        """
        Parameters
        ----------
        sc, fc : numpy.ndarray
            Connectivity matrices for each subject
            3D with shape: (number of subjects, resolution, resolution)
        euc_dist : numpy.ndarray
            Euclidean distance matrix
            2D with shape: (resolution, resolution)
        hubs : numpy.ndarray
            Array with the indexes of the hub nodes
        regions : numpy.ndarray
            Array with the region number each node is assigned to
        """

        n = len(sc)
        res = len(euc_dist)

        # Init vars
        triu = int(res * (res - 1) / 2)
        triu_i = np.triu_indices(res, 1)
        self.adj = torch.zeros((n, triu), dtype=torch.float).to(device)
        self.sp = torch.zeros((n, res, res), dtype=torch.float).to(device)
        self.pp = [None] * n
        self.sample_idx = [None] * n

        # Fill vars
        for i in range(n):
            brain = Brain(sc[i], fc[i], euc_dist, hubs=hubs, regions=regions)
            streamlines = brain.streamlines()
            node_str = brain.node_strength(weighted=True)
            is_target_node = brain.is_target_node
            is_target_region = brain.is_target_region
            is_hub = brain.hubs(binary=True)
            neighbour_just_visited_node = brain.neighbour_just_visited_node
            leave_non_target_region = brain.leave_non_target_region
            inter_regional_connections = brain.inter_regional_connections(weighted=False, distinct=True)
            prev_visited_region = brain.prev_visited_region
            fns = [
                lambda loc, nxt, prev_nodes, target: streamlines[loc,nxt],
                lambda loc, nxt, prev_nodes, target: node_str[nxt],
                lambda loc, nxt, prev_nodes, target: is_target_node(nxt, target),
                lambda loc, nxt, prev_nodes, target: is_target_region(nxt, target),
                lambda loc, nxt, prev_nodes, target: is_hub[nxt],
                lambda loc, nxt, prev_nodes, target: neighbour_just_visited_node(nxt, prev_nodes),
                lambda loc, nxt, prev_nodes, target: leave_non_target_region(loc, nxt, target),
                lambda loc, nxt, prev_nodes, target: inter_regional_connections[nxt],
                lambda loc, nxt, prev_nodes, target: prev_visited_region(loc, nxt, prev_nodes)]
            weights = list(np.random.random(size=len(fns)))
            self.adj[i] = T(brain.sc_bin[triu_i].astype(np.int), dtype=torch.int).to(device)
            self.sp[i] = T(brain.shortest_paths(), dtype=torch.float).to(device)
            self.pp[i] = PreferredPath(adj=brain.sc_bin, fn_vector=fns, fn_weights=weights)
            self.sample_idx[i] = np.column_stack(np.where(brain.fc_bin > 0))

    def __len__(self):
        return len(self.adj)

    def __getitem__(self, idx):
        return (self.adj[idx], self.sp[idx], self.pp[idx], self.sample_idx[idx])


class PolicyEstimator(Module):
    def __init__(self, res, fn_len, hidden_units=20):
        """
        Parameters
        ----------
        res : int
            Brain resolution (i.e. number of nodes)
        fn_len : int
            Number of function criteria
        hidden_units : int, optional
            Number of units in the hidden layer, by default 20
        """

        super(PolicyEstimator, self).__init__()
        self.n_inputs = int(res * (res - 1) / 2)
        self.n_outputs = fn_len * 2 # includes both mean and sigma
        self.network = Sequential(
            Linear(self.n_inputs, hidden_units),
            ReLU(),
            Linear(hidden_units, self.n_outputs))

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


def reinforce(pe, opt, data, epochs, batch, lr, sample=0, plt_data=None, plt_freq=0, plt_off=0, plt_avg=None, save_path=None, log=False):
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
    plt_freq : int
        How often to draw a plot of the model's progress
    plt_off : int
        Number of batches along the x-axis to offset when drawing the plots
    plt_avg : int
        Number of batches to average for the moving average lines in the plots (0 to not include moving average lines)
    save_path : str
        Location to save the state of the neural network and optimiser as well as plt_data
    log : bool
        Whether or not to continuously print the current epoch and batch
    """

    # Setup
    len_data = len(data)
    num_fns = data.pp[0].fn_length
    res = len(data.sp[0])
    plt_subtitle = f'\n(n={len_data}, res={res}, batch size={batch}, samples={"full" if sample == 0 else sample})'

    # Update learning rate
    for g in opt.param_groups:
        g['lr'] = lr

    # Epoch
    for _ in range(epochs):
        if log:
            e = plt_data['epochs']
            print(f'\r-- Epoch {e+1} --')
        epoch_fn(pe, opt, data, batch, sample, num_fns, plt_data, plt_freq, plt_off, plt_avg, plt_subtitle, log)

        # Save
        if save_path:
            save(save_path, pe, opt, plt_data)

        if log:
            print('\rDone')


def epoch_fn(pe, opt, data, batch, sample, num_fns, plt_data, plt_freq, plt_off, plt_avg, plt_subtitle, log):
    t1 = datetime.now() # Track epoch duration
    offset = 0
    while offset + batch <= len(data):
        rewards = torch.zeros((batch,1), dtype=torch.float).to(device)
        success = torch.zeros(batch, dtype=torch.float).to(device)
        adj, sp, pp, sample_idx = data[offset:offset+batch]
        probs = pe.predict(adj)
        mu, sig = probs[:,:num_fns], abs(probs[:,num_fns:]) + 1
        N = Normal(mu, sig)
        actions = N.sample().to(device)

        # Batch
        for i in range(batch):
            if log:
                print(f'\r{str(i+1+offset)}', end='')
            pp[i].fn_weights = actions[i].tolist()
            rewards[i], success[i] = sample_batch_fn(pp[i], sp[i], sample, sample_idx[i]) if sample > 0 else full_batch_fn(pp[i], sp[i])

        # Step
        step_fn(opt, N, actions, rewards)

        # Update plotting data
        if plt_data is not None:
            # Add data to arrays
            plt_data['rewards'].append(rewards.mean().item())
            plt_data['success'].append(success.mean())
            for j in range(num_fns):
                plt_data['mu'][j].append(mu[:,j].mean().item())
                plt_data['sig'][j].append(sig[:,j].mean().item())

            # Visualise
            #if plt_freq > 0:
            #    len_rewards = len(plt_data['rewards'])
            #    if (len_rewards + 1) % plt_freq == 0:
            #        plot(plt_data=plt_data, num_fns=num_fns, plt_avg=plt_avg, plt_off=plt_off, plt_subtitle=plt_subtitle)
            #        display.clear_output(wait=True)
            #        display.display(plt.gcf())

        # Run next batch
        offset += batch

    # Track epoch data
    t2 = datetime.now()
    plt_data['epoch_seconds'].append((t2-t1).seconds)
    plt_data['epochs'] += 1


def sample_batch_fn(pp, sp, sample, sample_idx):
    rewards = torch.zeros(sample, dtype=torch.float).to(device)
    success = torch.zeros(sample, dtype=torch.float).to(device)
    len_sample_idx = len(sample_idx)

    for i in range(sample):
        sp_val = 0
        while sp_val <= 0:
            s, t = sample_idx[np.random.choice(len_sample_idx)]
            sp_val = sp[s,t]
        pred = PreferredPath._single_path_formatted(pp._fwd, s, t, False)
        rewards[i] = local_reward(pred, sp_val)
        success[i] = pred != -1

    return rewards.mean(), success.mean()


def full_batch_fn(pp, sp):
    mask = torch.where(sp > 0).to(device)
    pred = pp.retrieve_all_paths()[mask]
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