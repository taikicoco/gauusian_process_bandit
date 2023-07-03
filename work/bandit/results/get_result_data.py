import pandas as pd

def get_record_count(df, f_max, f_sigma, algorithm):
    noise, beta, alpha, gp_me = 0.01, 0.01, 0.01, 0
    filtered_df = df[(df['f_max'] == f_max) & (df['f_sigma'] == f_sigma) 
                        & (df['noise'] == noise) & (df['beta'] == beta) 
                        & (df['alpha'] == alpha) & (df['gp_me'] == gp_me)]
    record_count = len(filtered_df)
    print(f"algorithm = {algorithm} f_max = {f_max} f_sigma = {f_sigma} Number of records = {record_count}")


if __name__ == '__main__':
    params = [(0.6, 0.3), (0.6, 0.1), (0.2, 0.3), (0.2, 0.1)] # [(f_max, f_sigma)]

    df_gp_ucb = pd.read_csv('./csv/gp_ucb/gp_ucb_2.csv')
    algorithm = 'gp_ucb'
    for f_max, f_sigma in params:
        get_record_count(df_gp_ucb, f_max, f_sigma, algorithm)

    df_gp_ts = pd.read_csv('./csv/gp_ts/gp_ts_2.csv')
    algorithm = 'gp_ts'
    for f_max, f_sigma in params:
        get_record_count(df_gp_ts, f_max, f_sigma, algorithm)
