import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

def make_parameter_analysis(df, f_max, f_sigma):
    lev = [i for i in range(0, 190, 10)]
    n_play = 300
    t_t_r = f_max * n_play
    
    # Noise-Beta Plot
    df = df
    df_subset1 = df[['noise', 'alpha', 'beta', 'total_reward']]
    df_subset1 = df_subset1.groupby(['noise', 'beta'])['total_reward'].mean().reset_index()
    
    pivot_table1 = df_subset1.pivot(index='noise', columns='beta', values='total_reward')
    pivot_table1 = pivot_table1.fillna(0)
    
    X1 = pivot_table1.columns.values
    Y1 = pivot_table1.index.values
    Z1 =  t_t_r - pivot_table1.values
    X1, Y1 = np.meshgrid(X1, Y1)
    
    # Alpha-Beta Plot
    df_subset2 = df[['noise', 'alpha', 'beta', 'total_reward']]
    df_subset2 = df_subset2.groupby(['alpha', 'beta'])['total_reward'].mean().reset_index()
    
    pivot_table2 = df_subset2.pivot(index='alpha', columns='beta', values='total_reward')
    pivot_table2 = pivot_table2.fillna(0)
    
    X2 = pivot_table2.columns.values
    Y2 = pivot_table2.index.values
    Z2 = t_t_r - pivot_table2.values
    X2, Y2 = np.meshgrid(X2, Y2)
    
    # Alpha-Noise Plot
    df_subset3 = df[['noise', 'alpha', 'beta', 'total_reward']]
    df_subset3 = df_subset3.groupby(['alpha', 'noise'])['total_reward'].mean().reset_index()
    
    pivot_table3 = df_subset3.pivot(index='alpha', columns='noise', values='total_reward')
    pivot_table3 = pivot_table3.fillna(0)
    
    X3 = pivot_table3.columns.values
    Y3 = pivot_table3.index.values
    Z3 = t_t_r - pivot_table3.values
    X3, Y3 = np.meshgrid(X3, Y3)
    
    # Create a figure
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot Noise-Beta
    c1 = axs[0].contour(X1, Y1, Z1,colors='black', levels=lev)
    c1.clabel(fmt='%1.1f', fontsize=8) 
    c1 = axs[0].contourf(X1, Y1, Z1,cmap='rainbow', levels=lev)  
    
    fig.colorbar(c1, ax=axs[0], label='total_reward')
    axs[0].set_xlabel('alpha')
    axs[0].set_ylabel('noise')
    axs[0].set_aspect('equal')
    
    # Plot Alpha-Beta
    c2 = axs[1].contour(X2, Y2, Z2,colors='black', levels=lev)
    c2.clabel(fmt='%1.1f', fontsize=8) 
    c2 = axs[1].contourf(X2, Y2, Z2,cmap='rainbow', levels=lev)  
    
    fig.colorbar(c2, ax=axs[1], label='total_reward')
    axs[1].set_xlabel('alpha')
    axs[1].set_ylabel('beta')
    axs[1].set_aspect('equal')
    
    # Plot Alpha-Noise
    
    c3 = axs[2].contour(X3, Y3, Z3,colors='black', levels=lev)
    c3.clabel(fmt='%1.1f', fontsize=8) 
    c3 = axs[2].contourf(X3, Y3, Z3,cmap='rainbow', levels=lev)  
    
    fig.colorbar(c3, ax=axs[2], label='total_reward')
    axs[2].set_xlabel('noise')
    axs[2].set_ylabel('beta')
    axs[2].set_aspect('equal')
    
    # Show the figure
    plt.tight_layout()
    plt.savefig(f"./img/gp-ts_parameter_{f_max}_{f_sigma}.png")

def save_parameter_analysis(f_max, f_sigma):
    df_gp_ts = pd.read_csv('./csv/gp_ts/gp_ts_2.csv')
    df = df_gp_ts[(df_gp_ts['f_max'] == f_max) & (df_gp_ts['f_sigma'] == f_sigma)]
    make_parameter_analysis(df, f_max, f_sigma)


if __name__ == '__main__':
    print("Start: make_gp_ts_param.py")
    f_sigma_list = [0.1, 0.3]
    f_max_list   = [0.2, 0.6]
    for f_sigma in f_sigma_list:
        for f_max in f_max_list:
            save_parameter_analysis(f_max, f_sigma)
    print("End: make_gp_ts_param.py")
