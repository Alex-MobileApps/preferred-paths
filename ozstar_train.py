import numpy as np
import torch
from scipy.io import loadmat
from ml import BrainDataset, PolicyEstimator, reinforce
from utils import device, train_cv_test_split
from argparse import ArgumentParser
from preferred_path import PreferredPath

if __name__ == "__main__":
    # Input Parameters
    parser = ArgumentParser()
    add = lambda arg, default, t, const=None: parser.add_argument(f'--{arg}', nargs='?', const=const, default=default, type=t)
    add_list = lambda arg: parser.add_argument(f'--{arg}', nargs='+')
    add('res', 219, int)
    add('subj', 0, int)
    add('epoch', 1, int)
    add('batch', 4, int)
    add('sample', 100, int)
    add('hu', 100, int)
    add('lr', 0.001, float)
    add('save', None, str)
    add('load', None, str)
    add('savefreq', 1, int)
    add('nolog', False, bool, const=True)
    add('pathmethod', PreferredPath._DEF_METHOD, str)
    add_list('fns')
    args = vars(parser.parse_args())

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
    save_freq = args['savefreq']
    log = not args['nolog']
    path_method = args['pathmethod']
    fns = args['fns']
    num_fns = len(fns)

    print('\n====================')
    subj_name = f'x{len(subj)}' if len(subj) > 1 else f's{str(subj[0] + 1).zfill(3)}'
    print(f'Running with parameters:', f'device = {device}', f'res = {res}', f'subj = {subj_name}', f'epochs = {epoch}', f'batch_size = {batch}', f'samples = {sample}', f'hidden_units = {hidden_units}', f'lr = {lr}', f'save_path = {save_path}', f'load_path = {load_path}', f'log_output = {log}', f'path_method = {path_method}', f'Functions = {num_fns} ({", ".join([f for f in fns])})', sep='\n')
    print('====================', flush=True)
    if log:
        print('Reading files...')

    # Read brain data
    sc = loadmat(f'/fred/oz192/data_n484/subjfiles_SC{res}.mat')
    fc = loadmat(f'/fred/oz192/data_n484/subjfiles_FC{res}.mat')
    sc = np.array([sc[f's{str(z+1).zfill(3)}'] for z in subj])
    fc = np.array([fc[f's{str(z+1).zfill(3)}'] for z in subj])
    euc_dist = loadmat('/fred/oz192/euc_dist.mat')[f'eu{res}']
    hubs = np.loadtxt(f'/fred/oz192/data_n484/hubs_{res}.txt', dtype=np.int, delimiter=',')
    regions = np.loadtxt(f'/fred/oz192/data_n484/regions_{res}.txt', dtype=np.int, delimiter=',')
    func_regions = np.loadtxt(f'/fred/oz192/data_n484/func_reg{res}.txt', dtype=np.int, delimiter=',')

    # Network parameters
    if log: print("Creating network...")
    pe = PolicyEstimator(res, num_fns, hidden_units=hidden_units).to(device)
    opt = torch.optim.Adam(pe.network.parameters(), lr=lr)

    # Init new/load previous training data
    if load_path:
        # Load from checkpoint
        plt_data = torch.load(load_path)
        pe.network.load_state_dict(plt_data.pop('model_state_dict'))
        opt.load_state_dict(plt_data.pop('optimizer_state_dict'))
        if plt_data['fns'] != fns: raise ValueError('Different function criteria loaded')
    else:
        # New
        plt_data = {
            'epochs': 0,
            'epoch_seconds': [],
            'rewards': [],
            'success': [],
            'mu': [[] for _ in range(num_fns)],
            'sig': [[] for _ in range(num_fns)],
            'fns': fns}
        plt_data['train_idx'], plt_data['cv_idx'], plt_data['test_idx'] = train_cv_test_split(subj, train_pct=0.6, cv_pct=0.2)

    # Train / test split
    if log: print('Loading brains...')
    train_idx = plt_data['train_idx']
    if len(train_idx) == 1: train_idx = [0]
    train_data = BrainDataset(sc[train_idx], fc[train_idx], euc_dist, hubs, regions, func_regions, fns)
    if log: print('====================')

    # Reinforce and save after each epoch
    reinforce(pe, opt, train_data, epochs=epoch, batch=batch, sample=sample, lr=lr, plt_data=plt_data, save_path=save_path, save_freq=save_freq, log=log, path_method=path_method)