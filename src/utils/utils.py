import gc
import random
import numpy as np
from scipy.stats import zscore
import torch


def clear_cache():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    gc.collect()


def config(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def string_join(x, j=''):
    return j.join(x)


def find_outliers(df, field='time', transform=lambda x: x, ignore_zeros=False):
    if ignore_zeros:
        df['temp_consider_rows'] = df[field] != 0
        df.loc[~df['temp_consider_rows'], 'outlier'] = False
    else:
        df['temp_consider_rows'] = True

    z_scores = zscore(transform(df.loc[df['temp_consider_rows'], field]))
    abs_z_scores = np.abs(z_scores)

    df.loc[df['temp_consider_rows'], 'outlier'] = abs_z_scores > 3
    print("Percentage of outliers:", sum(df['outlier']) / len(df))

    del df['temp_consider_rows']
    return df


def fnames(dataset_name, model_name):
    df_name = f'checkpoints/{dataset_name}_{model_name}_df.tsv'
    return df_name


def write_tsv(df, fname):
    df.to_csv(fname, sep='\t')
