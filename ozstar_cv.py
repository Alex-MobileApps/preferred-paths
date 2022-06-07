import numpy as np
import torch
from scipy.io import loadmat
from ml import BrainDataset, sample_batch_fn, device
from argparse import ArgumentParser

if __name__ == "__main__":
    # Input Parameters
    parser = ArgumentParser()
    add = lambda arg, default, t, const=None: parser.add_argument(f'--{arg}', nargs='?', const=const, default=default, type=t)
    add('res', 219, int)
    add('batch', 1, int)
    add('sample', 0, int)
    add('save', None, str)
    add('load', None, str)
    add('nolog', False, bool, const=True)
    args = vars(parser.parse_args())

    # Read arguments
    res = args['res']
    batch = args['batch']
    sample = args['sample']
    load_path = args['load']
    save_path = args['save']
    log = not args['nolog']

    if log:
        print('\n====================')
        print(f'Running with parameters:', f'res = {res}', f'batch_size = {batch}', f'samples = {sample}', f'save_path = {save_path}', f'load_path = {load_path}', f'log_output = {log}', sep='\n')
        print('====================')
        print('Reading files...')

    # Read train data file
    plt_data = torch.load(load_path, map_location=device)
    num_fns = len(plt_data['mu'])
    num_train = len(plt_data['train_idx'])
    epochs = plt_data['epochs']
    num_avg = int(num_train / batch)

    # Train data epoch rewards
    if log: print("Generating per epoch training rewards...")
    train_rewards = np.zeros(epochs)
    train_mu = np.zeros((num_fns, epochs))
    off = 0
    for i in range(epochs):
        train_rewards[i] = sum(plt_data['rewards'][off:off+num_avg]) / num_avg
        for j in range(num_fns):
            train_mu[j,i] = sum(plt_data['mu'][j][off:off+num_avg]) / num_avg
        off += num_avg

    # Generate CV brains
    if log: print("Generating cv brains...")
    subj = plt_data['cv_idx']

    sc = loadmat(f'/fred/oz192/data_n484/subjfiles_SC{res}.mat')
    fc = loadmat(f'/fred/oz192/data_n484/subjfiles_FC{res}.mat')
    sc = np.array([sc[f's{str(z+1).zfill(3)}'] for z in subj])
    fc = np.array([fc[f's{str(z+1).zfill(3)}'] for z in subj])
    euc_dist = loadmat('/fred/oz192/euc_dist.mat')[f'eu{res}']
    hubs = np.loadtxt(f'/fred/oz192/data_n484/hubs_{res}.txt', dtype=int, delimiter=',')
    regions = np.loadtxt(f'/fred/oz192/data_n484/regions_{res}.txt', dtype=int, delimiter=',')

    cv_data = BrainDataset(sc, fc, euc_dist, hubs, regions)

    # CV data epoch rewards
    if log:
        print("Generating per epoch cv rewards...")
        print('====================')
    cv_rewards = np.zeros(epochs)
    for i in range(epochs):
        if log: print (f"--- Epoch {i+1} ---")
        fn_weights = train_mu[:,i]
        tmp_cv_rewards = np.zeros(len(cv_data))
        for j, (_, sp, pp, sample_idx) in enumerate(cv_data):
            if log: print(f"\r{j+1}", end='')
            pp.fn_weights = fn_weights
            tmp_cv_rewards[j], _ = sample_batch_fn(pp, sp, sample, sample_idx)
        cv_rewards[i] = tmp_cv_rewards.mean()
        if log: print("\rDone")

    train_rewards = list(train_rewards)
    cv_rewards = list(cv_rewards)

    if log:
        print("=============")
        print("---  Train rewards per epoch --- ")
        print(train_rewards)
        print("\n---  CV rewards per epoch --- ")
        print(cv_rewards)
        print("=============")

    # Save
    if save_path:
        save_data = {
            'train_epoch_rewards': train_rewards,
            'cv_epoch_rewards': cv_rewards}
        torch.save(save_data, save_path)
