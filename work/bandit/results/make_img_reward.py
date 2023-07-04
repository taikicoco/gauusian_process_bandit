import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def calculate_statistics(df, df2, f_max, f_sigma, algorithm):
    noise, alpha, beta, gp_me = find_best_params(df2, f_max, f_sigma)
    df_subset = df[(df['f_max'] == f_max) & (df['f_sigma'] == f_sigma)]
    df_sub = df_subset[(df_subset['noise'] == noise) & (df_subset['beta'] == beta) 
                        & (df_subset['alpha'] == alpha) & (df_subset['gp_me'] == gp_me)]
    
    df_grouped = df_sub.groupby('play_time').agg({'reward': ['mean', 'var']})
    df_grouped['reward', 'mean'] = f_max - df_grouped['reward', 'mean'] 
    cumulative_reward_mean = df_grouped['reward', 'mean'].rolling(window=100).mean()
    cumulative_reward_std = np.sqrt(df_grouped['reward', 'var'].rolling(window=100).var())
    print(f'{algorithm} : alpha = {noise} beta = {alpha} sigma = {noise} gp_me = {gp_me}')
    
    return cumulative_reward_mean, cumulative_reward_std

def find_best_params(df, f_max, f_sigma):
    df_subset = df[(df['f_max'] == f_max) & (df['f_sigma'] == f_sigma)]
    df_subset = df_subset.groupby(['noise', 'alpha', 'beta', 'gp_me']).mean()
    max_reward = df_subset['total_reward'].max()
    max_reward_records = df_subset[df_subset['total_reward'] == max_reward].reset_index()
    return max_reward_records['noise'].values[0], max_reward_records['alpha'].values[0], max_reward_records['beta'].values[0], max_reward_records['gp_me'].values[0]

def plot_cumulative_reward_comparison(df_ts, df2_ts, df_ucb, df2_ucb, f_max, f_sigma):
    cumulative_reward_mean_ts, cumulative_reward_std_ts = calculate_statistics(df_ts, df2_ts, f_max, f_sigma, "GP-TS")
    cumulative_reward_mean_ucb, cumulative_reward_std_ucb = calculate_statistics(df_ucb, df2_ucb, f_max, f_sigma, "GP-UCB")

    le = [i for i in range(300)]
    
    # 95%信頼区間の計算
    confidence_interval_lower_ucb = cumulative_reward_mean_ucb -  cumulative_reward_std_ucb *2
    confidence_interval_upper_ucb = cumulative_reward_mean_ucb +  cumulative_reward_std_ucb *2
    
    confidence_interval_lower_ts = cumulative_reward_mean_ts -  cumulative_reward_std_ts * 2
    confidence_interval_upper_ts = cumulative_reward_mean_ts +  cumulative_reward_std_ts * 2

    plt.figure()
    plt.plot(le, cumulative_reward_mean_ucb, label='GP-UCB Mean')
    plt.plot(le, cumulative_reward_mean_ts, label='GP-TS Mean')
    plt.axhline(0, color='red', linestyle='dashed', alpha=0.5)
    plt.fill_between(le, confidence_interval_lower_ucb, confidence_interval_upper_ucb, alpha=0.2, label='UCB 95% Confidence Interval')
    plt.fill_between(le, confidence_interval_lower_ts, confidence_interval_upper_ts, alpha=0.2, label='GP-TS 95% Confidence Interval')
    plt.xlabel('Play Time')
    plt.ylabel('Reglet')
    plt.legend()
    plt.grid(True)  
    plt.ylim(-0.15,0.5)
    plt.xlim(100,300)
    plt.savefig(f"./img/gp-reward_{f_max}_{f_sigma}.png")
    print(f"save ./img/gp-reward_{f_max}_{f_sigma}.png")
    plt.close()


if __name__ == '__main__':
    df2_ts = pd.read_csv('./csv/gp_ts/gp_ts_2.csv')
    df_ts = pd.read_csv('./csv/gp_ts/gp_ts.csv')

    df2_ucb = pd.read_csv('./csv/gp_ucb/gp_ucb_2.csv')
    df_ucb = pd.read_csv('./csv/gp_ucb/gp_ucb.csv')

    f_sigma_list = [0.1, 0.3]
    f_max_list   = [0.2, 0.6]

    for f_max in f_max_list:
        for f_sigma in f_sigma_list:
            plot_cumulative_reward_comparison(df_ts, df2_ts, df_ucb, df2_ucb, f_max, f_sigma)
    

