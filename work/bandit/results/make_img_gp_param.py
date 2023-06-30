import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def make_loss_contour_plot(fig, ax, df, x_col, y_col, lev, t_t_r, f_max, f_sigma):
    df_subset = df[[x_col, y_col, 'total_reward']]
    df_subset = df_subset.groupby([x_col, y_col])['total_reward'].mean().reset_index()

    pivot_table = df_subset.pivot(index=x_col, columns=y_col, values='total_reward').fillna(0)

    X = pivot_table.columns.values
    Y = pivot_table.index.values
    Z = t_t_r - pivot_table.values
    X, Y = np.meshgrid(X, Y)

    c = ax.contour(X, Y, Z, colors='black', levels=lev)
    c.clabel(fmt='%1.1f', fontsize=8)
    c = ax.contourf(X, Y, Z, cmap='rainbow', levels=lev)

    fig.colorbar(c, ax=ax, label='total_reward')
    ax.set_xlabel(y_col)
    ax.set_ylabel(x_col)
    ax.set_aspect('equal')
    ax.set_title(f'{x_col}-{y_col} Plot\n f_max={f_max}, f_sigma={f_sigma}')

def make_parameter_analysis(df, f_max, f_sigma, algorithm):
    lev = [i for i in range(0, 190, 10)]
    n_play = 300
    t_t_r = f_max * n_play

    param_combinations = [('noise', 'gp_me'), ('noise', 'alpha'), ('noise', 'beta'),
                            ('gp_me', 'alpha'), ('gp_me', 'beta'), ('alpha', 'beta')]

    fig, axs = plt.subplots(2, 3, figsize=(15, 10))  # Changed from 1 row to 2 rows and 3 columns

    for i, (x_col, y_col) in enumerate(param_combinations):
        row, col = divmod(i, 3)  # Calculate row and column for subplot
        make_loss_contour_plot(fig, axs[row, col], df, x_col, y_col, lev, t_t_r, f_max, f_sigma)

    plt.tight_layout()

    if algorithm == 'gp_ucb':
        plt.savefig(f"./img/gp-ucb_parameter_{f_max}_{f_sigma}.png")
    elif algorithm == 'gp_ts':
        plt.savefig(f"./img/gp-ts_parameter_{f_max}_{f_sigma}.png")

def save_parameter_analysis(df_gp_ts, f_max, f_sigma, algorithm):
    df = df_gp_ts[(df_gp_ts['f_max'] == f_max) & (df_gp_ts['f_sigma'] == f_sigma)]
    make_parameter_analysis(df, f_max, f_sigma, algorithm)

if __name__ == '__main__':

    f_sigma_list = [0.1, 0.3]
    f_max_list   = [0.2, 0.6]

    print("Start: make_gp_ucb_param.py")
    df_gp_ucb = pd.read_csv('./csv/gp_ucb/gp_ucb_2.csv')
    algorithm = 'gp_ucb'
    for f_sigma in f_sigma_list:
        for f_max in f_max_list:
            save_parameter_analysis(df_gp_ucb, f_max, f_sigma, algorithm)
    print("End: make_gp_ucb_param.py")

    print("Start: make_gp_ts_param.py")
    df_gp_ts = pd.read_csv('./csv/gp_ts/gp_ts_2.csv')
    algorithm = 'gp_ts'
    for f_sigma in f_sigma_list:
        for f_max in f_max_list:
            save_parameter_analysis(df_gp_ts, f_max, f_sigma, algorithm)
    print("End: make_gp_ts_param.py")
