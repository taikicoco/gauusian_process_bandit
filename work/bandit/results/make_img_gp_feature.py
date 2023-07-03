import matplotlib.pyplot as plt
import pandas as pd

def filter_and_calculate(df, f_max, f_sigma):
    df_subset = df[(df['f_sigma'] == f_sigma) & (df['f_max'] == f_max)]
    df_subset = df_subset.groupby(['noise', 'alpha', 'beta', 'gp_me']).mean()
    df_subset['total_reward'] = f_max*300 - df_subset['total_reward']
    return df_subset.reset_index()

def calculate_mean_std(df, feature):
    mean = df.groupby(feature)['total_reward'].mean()
    std = df.groupby(feature)['total_reward'].std()
    return mean, std

def plot_parameter_loss_reward(algorithm, dfs, feature, labels, colors, fmts):
    fig, ax = plt.subplots(figsize=(8, 6))

    for df, label, color, fmt in zip(dfs, labels, colors, fmts):
        mean, std = calculate_mean_std(df, feature)
        ax.errorbar(mean.index, mean.values, yerr=std.values, fmt=fmt, color=color, elinewidth=5, ecolor=(0.8, 0, 0, 0.5), label=label)

    plt.ylim(0, 190)
    ax.set_xlabel(feature)
    ax.set_ylabel('Regret')
    ax.grid(True)
    ax.legend()

    plt.tight_layout()
    if algorithm == 'gp_ucb':
        plt.savefig(f"./img/gp-ucb-feature_{feature}.png")
        print(f"Saved ./img/gp-ucb-feature_{feature}.png")
    elif algorithm == 'gp_ts':
        plt.savefig(f"./img/gp-ts-feature_{feature}.png")
        print(f"Saved ./img/gp-ts-feature_{feature}.png")

if __name__ == '__main__':
    params = [(0.6, 0.3), (0.6, 0.1), (0.2, 0.3), (0.2, 0.1)] # [(f_max, f_sigma)]

    labels = [f'f_sigma = {param[1]}, f_max = {param[0]}' for param in params]
    colors = ['red', 'blue', 'green', 'purple']
    fmts = ['-o', '-v', '-s', '-D']
    features = ['noise', 'alpha', 'beta', 'gp_me']

    df_gp_ucb = pd.read_csv('./csv/gp_ucb/gp_ucb_2.csv')
    algorithm = 'gp_ucb'
    df_ucb_list = [filter_and_calculate(df_gp_ucb, f_max, f_sigma) for f_max, f_sigma in params]
    for feature in features:
        plot_parameter_loss_reward(algorithm, df_ucb_list, feature, labels, colors, fmts)

    df_gp_ts = pd.read_csv('./csv/gp_ts/gp_ts_2.csv')
    algorithm = 'gp_ts'
    df_ts_list = [filter_and_calculate(df_gp_ts, f_max, f_sigma) for f_max, f_sigma in params]    
    for feature in features:
        plot_parameter_loss_reward(algorithm, df_ts_list, feature, labels, colors, fmts)
