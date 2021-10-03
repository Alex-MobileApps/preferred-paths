import numpy as np
import torch
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
from ml import BrainDataset, PolicyEstimator, reinforce
from sys import exit
from argparse import ArgumentParser

if __name__ == "__main__":
    # Input Parameters
    parser = ArgumentParser()
    add = lambda arg, default, t: parser.add_argument(f'--{arg}', nargs='?', default=default, type=t)
    add('res', 219, int)
    add('subj', 0, int)
    add('epoch', 1, int)
    add('batch', 1, int)
    add('sample', 0, int)
    add('hu', 2000, int)
    add('lr', 0.005, float)
    add('save', None, str)
    add('load', None, str)
    add('log', True, bool)
    args = vars(parser.parse_args())

    num_fns = 9
    res = args['res']
    subj = args['subj']
    subj = list(range(484)) if subj == 0 else [subj-1]
    epoch = args['epoch']
    batch = args['batch']
    sample = args['sample']
    hidden_units = args['hu']
    lr = args['lr']
    save_path = args['save']
    load_path = args['load']
    log = args['log']

    # Confirm inputs before running
    print('\n====================')
    subj_name = f'x{len(subj)}' if len(subj) > 1 else f's{str(subj[0] + 1).zfill(3)}'
    print(f'Running with parameters:', f'res = {res}', f'subj = {subj_name}', f'epochs = {epoch}', f'batch_size = {batch}', f'samples = {sample}', f'hidden_units = {hidden_units}', f'lr = {lr}', f'save_path = {save_path}', f'load_path = {load_path}', f'log_output = {log}', sep='\n')
    value = input('Continue? (y/n): ')
    if value != '' and value.lower() != 'y':
        exit()
    print('====================')

    # Read brain data
    print('Reading files...')
    sc = loadmat(f'data/subjfiles_SC{res}.mat')
    fc = loadmat(f'data/subjfiles_FC{res}.mat')
    sc = np.array([sc[f's{str(z+1).zfill(3)}'] for z in subj])
    fc = np.array([fc[f's{str(z+1).zfill(3)}'] for z in subj])
    euc_dist = loadmat('data/euc_dist.mat')[f'eu{res}']
    hubs = np.loadtxt(f'data/hubs_{res}.txt', dtype=np.int, delimiter=',')
    regions = np.loadtxt(f'data/regions_{res}.txt', dtype=np.int, delimiter=',')

    # Network parameters
    pe = PolicyEstimator(res, num_fns)
    opt = torch.optim.Adam(pe.network.parameters(), lr=lr)

    # Init new/load previous training data
    if load_path:
        # Load from checkpoint
        checkpoint = torch.load(load_path)
        plt_data = {k: checkpoint[k] for k in ('rewards','success','mu','sig','train_idx','test_idx')}
        pe.network.load_state_dict(checkpoint['model_state_dict'])
        opt.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        # New
        plt_data = {
            'rewards': [],
            'success': [],
            'mu': [[] for _ in range(num_fns)],
            'sig': [[] for _ in range(num_fns)]}
        plt_data['train_idx'], plt_data['test_idx'] = train_test_split(subj, train_size=0.7) if len(subj) > 1 else (subj, [])

    # Train / test split
    print('Loading brains...')
    train_idx, test_idx = plt_data['train_idx'], plt_data['test_idx']
    train_data = BrainDataset(sc[train_idx], fc[train_idx], euc_dist, hubs, regions)
    test_data =  BrainDataset(sc[test_idx],  fc[test_idx],  euc_dist, hubs, regions)
    print('====================')

    # Reinforce and save after each epoch
    reinforce(pe, opt, train_data, epochs=epoch, batch=batch, sample=sample, lr=lr, plt_data=plt_data, save_path=save_path, log=log)