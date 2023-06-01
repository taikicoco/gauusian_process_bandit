import os
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
import pandas as pd

import sys
sys.path.append('./models') 
from gaussian import GP
from gaussian import RBF
from bandit import Bandit

sys.path.append('./train_data') 
from train_func import TrueFunc


# シミュレーションのパラメータ設定
####################
n_play = 300
n_sample = 100

func_sigma_list = [0.1, 0.3]
func_max_list   = [0.2, 0.6]

gp_noise_list   = [i/100 for i in range(5, 35, 5)]
gp_me_list      = [0.5]

rbf_alpha_list  = [i/100 for i in range(5, 35, 5)]
rbf_beta_list   = [i/100 for i in range(5, 35, 5)] 
####################

def main_sim(t):
    T = t
    for _ in range(T):
        play_bandit_gp_ucb()


def get_max_seed_ucb():
    try:
        d = pd.read_csv('../results/csv/gp_ucb/gp_ucb.csv')
        max_seed = d['seed'].tail(1).values[0]
        return  max_seed
    except FileNotFoundError:
        return 0
    

def play_bandit_gp_ucb():
    
    dir_path = '../results/csv/gp_ucb/'
    column   =['f_sigma', 'f_max', 'noise', 'gp_me',  'alpha','beta','seed', 'play_time', 'select_arm', 'reward', ]
    column2  =['f_sigma', 'f_max', 'noise', 'gp_me', 'alpha','beta', 'total_reward', 'seed' ]
    
    data  = []
    data2 = []
    seed     = get_max_seed_ucb() + 1
    
    csv_path = '../results/csv/gp_ucb/gp_ucb.csv'
    csv_path2 = '../results/csv/gp_ucb/gp_ucb_2.csv'
    
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
                                train, reward = bandit.gp_ucb_e(train_func, model)
                                t_r += reward
                                data.append([train_func_sigma, train_func_max, gp_noise, gp_me, rbf_alpha, rbf_beta, seed, play_time, train, reward])
                            data2.append([train_func_sigma, train_func_max, gp_noise, gp_me, rbf_alpha, rbf_beta, t_r, seed])
                            
                            
    df = pd.DataFrame(data, columns=column)
    df2 = pd.DataFrame(data2, columns=column2)
    
    df = pd.concat([_df, df], ignore_index=True)
    df2 = pd.concat([_df2, df2], ignore_index=True)
    
    csv_name = "gp_ucb.csv"
    csv_name2 = "gp_ucb_2.csv"
    
    df.to_csv(str(dir_path) + str(csv_name), index=False)
    df2.to_csv(str(dir_path) + str(csv_name2), index=False)

    print(f'finished seed is {seed}')

main_sim(1)