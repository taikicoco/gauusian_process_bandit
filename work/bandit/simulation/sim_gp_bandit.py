import os
import numpy as np
import pandas as pd

from models.gaussian import GP
from models.gaussian import RBF
from models.bandit import Bandit
from train_data.train_func import TrueFunc

# シミュレーションのパラメータ設定
####################
n_play = 10
n_sample = 100

func_sigma_list = [0.1, 0.3]
func_max_list   = [0.2, 0.6]

log_space_array = np.geomspace(0.01, 0.5, 2)
log_space_array = list(np.round(log_space_array, decimals=2))
# log_space_array = [0.01 0.02 0.05 0.1  0.23 0.5 ]

gp_me_list      = [0, 0.5]
gp_noise_list   = log_space_array

rbf_alpha_list  = log_space_array
rbf_beta_list   = log_space_array
####################

def get_max_seed(algorithm):
    try:
        d = pd.read_csv(f'../results/csv/{algorithm}/{algorithm}.csv')
        max_seed = d['seed'].tail(1).values[0]
        return  max_seed
    except FileNotFoundError:
        return 0

def play_bandit_algorithm(algorithm):
    
    dir_path = f'../results/csv/{algorithm}/'
    column   =['f_sigma', 'f_max', 'noise', 'gp_me',  'alpha','beta','seed', 'play_time', 'select_arm', 'reward']
    column2  =['f_sigma', 'f_max', 'noise', 'gp_me', 'alpha','beta', 'total_reward', 'seed']
    
    data  = []
    data2 = []
    seed  = get_max_seed(algorithm) + 1
    
    csv_name = f"{algorithm}.csv"
    csv_name2 = f"{algorithm}_2.csv"
    csv_path = f'../results/csv/{algorithm}/{algorithm}.csv'
    csv_path2 = f'../results/csv/{algorithm}/{algorithm}_2.csv'
    
    
    if os.path.exists(csv_path):
        _df = pd.read_csv(csv_path)
    else:
        _df = pd.DataFrame(columns=column)
        
    if os.path.exists(csv_path2):
        _df2 = pd.read_csv(csv_path2)
    else:
        _df2 = pd.DataFrame(columns=column2)
    
    for train_func_sigma in func_sigma_list:
        for train_func_max in func_max_list:
            for gp_noise in gp_noise_list:

                print("#######################")
                print(f'algorithm = {algorithm}')
                print(f'gp_noise = {gp_noise}')
                print(f'train_func_sigma = {train_func_sigma}')
                print(f'train_func_max = {train_func_max}')

                for gp_me in gp_me_list:
                    for rbf_alpha in rbf_alpha_list:
                        for rbf_beta in rbf_beta_list:
                            train_func = TrueFunc(train_func_sigma, train_func_max).peak_one
                            kernel = RBF(rbf_alpha, rbf_beta)
                            model = GP(gp_me, gp_noise, kernel)

                            t_r = 0
                            np.random.seed(seed)
                            bandit = Bandit(n_play, n_sample)

                            for play_time in range(n_play):
                                train, reward = bandit.gp_ts_e(train_func, model)
                                t_r += reward
                                data.append([train_func_sigma, train_func_max, gp_noise, gp_me, rbf_alpha, rbf_beta, seed, play_time, train, reward])
                            data2.append([train_func_sigma, train_func_max, gp_noise, gp_me, rbf_alpha, rbf_beta, t_r, seed])
                            
                            
    df = pd.DataFrame(data, columns=column)
    df = pd.concat([_df, df], ignore_index=True)
    df.to_csv(str(dir_path) + str(csv_name), index=False)

    df2 = pd.DataFrame(data2, columns=column2)
    df2 = pd.concat([_df2, df2], ignore_index=True)
    df2.to_csv(str(dir_path) + str(csv_name2), index=False)

    print(f'finished seed is {seed}')

if __name__ == '__main__':
    traial_number = 1
    for _ in range(traial_number):
        play_bandit_algorithm('gp_ts')
    for _ in range(traial_number):
        play_bandit_algorithm('gp_ucb')