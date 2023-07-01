import os
import numpy as np
import pandas as pd

from models.gaussian import GP
from models.gaussian import RBF
from models.bandit import Bandit
from train_data.train_func import TrueFunc

import google_chat_notifier

# Parameters
PARAMS = {
    'n_play': 300,
    'n_sample': 100,
    'func_sigma_list': [0.1, 0.3],
    'func_max_list': [0.2, 0.6],
    'gp_me_list': [0, 0.5, 1.0],
    'gp_noise_list': np.round(np.geomspace(0.01, 0.5, 8), decimals=2).tolist(),
    'rbf_alpha_list': np.round(np.geomspace(0.01, 0.5, 8), decimals=2).tolist(),
    'rbf_beta_list': np.round(np.geomspace(0.01, 0.5, 8), decimals=2).tolist()
}

# Get the maximum seed
def get_max_seed(algorithm):
    try:
        df = pd.read_csv(f'../results/csv/{algorithm}/{algorithm}.csv')
        return df['seed'].tail(1).values[0] + 1
    except FileNotFoundError:
        return 1

# Generate the simulation data
def generate_data(seed, algorithm):
    data = []
    data2 = []
    for train_func_sigma in PARAMS['func_sigma_list']:
        for train_func_max in PARAMS['func_max_list']:
            for gp_me in PARAMS['gp_me_list']:

                print("##############################")
                print(f"algorithm = {algorithm}")
                print(f"train_func_sigma = {train_func_sigma}")
                print(f"train_func_max = {train_func_max}")
                print(f"gp_me = {gp_me}")

                for gp_noise in PARAMS['gp_noise_list']:        
                    for rbf_alpha in PARAMS['rbf_alpha_list']:
                        for rbf_beta in PARAMS['rbf_beta_list']:
                            train_func = TrueFunc(train_func_sigma, train_func_max).peak_one
                            kernel = RBF(rbf_alpha, rbf_beta)
                            model = GP(gp_me, gp_noise, kernel)
                            t_r = 0
                            np.random.seed(seed)
                            bandit = Bandit(PARAMS['n_play'], PARAMS['n_sample'])

                            for play_time in range(PARAMS['n_play']):
                                if algorithm == 'gp_ucb':
                                    train, reward = bandit.gp_ucb_e(train_func, model)
                                elif algorithm == 'gp_ts':
                                    train, reward = bandit.gp_ts_e(train_func, model)
                                t_r += reward
                                data.append([train_func_sigma, train_func_max, gp_noise, gp_me, rbf_alpha, rbf_beta, seed, play_time, train, reward])
                            data2.append([train_func_sigma, train_func_max, gp_noise, gp_me, rbf_alpha, rbf_beta, t_r, seed])

    return data, data2

# Save the data to csv
def save_to_csv(algorithm, data, column_names, suffix=''):
    dir_path = f'../results/csv/{algorithm}/'
    os.makedirs(dir_path, exist_ok=True)
    df_path = dir_path + f"{algorithm}{suffix}.csv"
    df = pd.DataFrame(data, columns=column_names)
    
    if os.path.exists(df_path):
        df_existing = pd.read_csv(df_path)
        df = pd.concat([df_existing, df])
    
    df.to_csv(df_path, index=False)

if __name__ == '__main__':
    algorithms = ['gp_ts', 'gp_ucb']
    column_names1 = ['f_sigma', 'f_max', 'noise', 'gp_me', 'alpha', 'beta', 'seed', 'play_time', 'select_arm', 'reward']
    column_names2 = ['f_sigma', 'f_max', 'noise', 'gp_me', 'alpha', 'beta', 'total_reward', 'seed']
    
    Traial = 1
    for _ in range(Traial):
        for algorithm in algorithms:
            seed = get_max_seed(algorithm)
            data1, data2 = generate_data(seed, algorithm)
            save_to_csv(algorithm, data1, column_names1)
            save_to_csv(algorithm, data2, column_names2, '_2')
            print(f'Finished Bandit algo =  {algorithm} seed =  {seed}')

    google_chat_notifier.send_message_to_google_chat("Finished Bandit simulation")
