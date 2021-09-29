import numpy as np
from brain import Brain
from preferred_path import PreferredPath
import matplotlib.pyplot as plt
import torch
from torch.nn import Sequential, Linear, ReLU
from torch.distributions import Normal
from IPython import display
from cust_plot import plot


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
        self.adj = np.zeros((n, triu))
        self.sp = np.zeros((n, res, res))
        self.pp = [None] * n

        # Fill vars
        for i in range(n):
            brain = Brain(sc[i], fc[i], euc_dist, hubs=hubs, regions=regions)
            streamlines = brain.streamlines()
            node_str = brain.node_strength(weighted=False)
            is_target_node = brain.is_target_node
            is_target_region = brain.is_target_region
            is_hub = brain.hubs(binary=True)
            neighbour_just_visited_node = brain.neighbour_just_visited_node
            fns = [
                lambda loc, nxt, prev_nodes, target: streamlines[loc,nxt],
                lambda loc, nxt, prev_nodes, target: node_str[nxt],
                lambda loc, nxt, prev_nodes, target: is_target_node(nxt, target),
                lambda loc, nxt, prev_nodes, target: is_target_region(nxt, target),
                lambda loc, nxt, prev_nodes, target: is_hub[nxt],
                lambda loc, nxt, prev_nodes, target: neighbour_just_visited_node(nxt, prev_nodes)]
            weights = list(np.random.random(size=len(fns)))
            self.adj[i] = brain.sc_bin[triu_i]
            self.sp[i] = brain.shortest_paths()
            self.pp[i] = PreferredPath(adj=brain.sc_bin, fn_vector=fns, fn_weights=weights)

    def __len__(self):
        return len(self.adj)

    def __getitem__(self, idx):
        return (self.adj[idx], self.sp[idx], self.pp[idx])


class PolicyEstimator():
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

        self.n_inputs = int(res * (res - 1) / 2)
        self.n_outputs = fn_len * 2 # includes both mean and ln(sigma)
        self.network = Sequential(
            Linear(self.n_inputs, hidden_units),
            ReLU(),
            Linear(hidden_units, self.n_outputs))

    def predict(self, state):
        return self.network(torch.FloatTensor(state))


def reward(pred, sp):
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
    local = np.zeros(sp.size)
    mask = np.where(pred != -1)
    local[mask] = sp[mask] / pred[mask]

    # Global navigation efficiency ratio
    return local.sum() / sp.size


def reinforce(pe, opt, data, epochs, batch, lr, plt_data=None, inc_plt=True, plt_freq=338, plt_off=0, plt_avg=None, save_path=None):
    # Setup
    len_data = len(data)
    num_fns = data.pp[0].fn_length
    res = len(data.sp[0])
    plt_notes = f'\n(n={len_data}, res={res}, batch size={batch})' # For plotting an extra note after the title

    # Update learning rate
    for g in opt.param_groups:
        g['lr'] = lr

    # Run
    for e in range(epochs):
        print(f'\r-- Epoch {e+1} --')
        offset = 0

        # Epoch
        while offset + batch <= len_data:
            rewards = torch.zeros(batch,1)
            success = np.zeros(batch)
            adj, sp, pp = data[offset:offset+batch]
            probs = pe.predict(adj)
            mu, sig = probs[:,:num_fns], abs(probs[:,num_fns:]) + 1
            m = Normal(mu, sig)
            actions = m.sample()

            # Batch
            for i in range(batch):
                print(f'\r{str(i+1+offset)}', end='')
                pp[i].fn_weights = actions[i].tolist()
                mask = np.where(sp[i] > 0)
                pred = pp[i].retrieve_all_paths()[mask]
                rewards[i] = reward(pred, sp[i][mask])
                success[i] = 1 - (pred == -1).sum() / len(pred)

            # Step
            opt.zero_grad()
            loss = -m.log_prob(actions) * rewards
            loss = loss.mean() - 0.5
            loss.backward()
            opt.step()

            if plt_data is not None:
                # Add data to arrays
                plt_data['rewards'].append(rewards.mean().item())
                plt_data['success'].append(success.mean())
                for j in range(num_fns):
                    plt_data['mu'][j].append(mu[:,j].mean().item())
                    plt_data['sig'][j].append(sig[:,j].mean().item())

                if inc_plt:
                    len_rewards = len(plt_data['rewards'])
                    if (len_rewards + 1) % plt_freq == 0:
                        plot(plt_data=plt_data, num_fns=num_fns, plt_avg=plt_avg, plt_off=plt_off, plt_notes=plt_notes)
                        display.clear_output(wait=True)
                        display.display(plt.gcf())

            # Run next batch
            offset += batch

        # Save at end of epoch
        print('\rDone')
        if save_path:
            save(save_path, pe, opt, plt_data)


def save(path, pe, opt, plt_data):
    save_data = {
        'model_state_dict': pe.network.state_dict(),
        'optimizer_state_dict': opt.state_dict()}
    for key, value in plt_data.items():
        save_data[key] = value
    torch.save(save_data, path)