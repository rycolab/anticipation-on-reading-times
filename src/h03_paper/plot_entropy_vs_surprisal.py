import os
import sys
import argparse
import pandas as pd
import scipy

import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import constants
from utils import plot as utils_plot



def get_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--datasets', type=str, nargs='+', default=constants.DATASETS)
    parser.add_argument('--model', type=str, default='gpt-small')

    return parser.parse_args()

def get_data(model, dataset):
    fname = f'checkpoints/rt_and_entropy/rt_vs_entropy-{dataset}-{model}.tsv'
    df = pd.read_csv(fname, sep='\t')
    df = df.groupby(['new_ind', 'text_id', 'sentence_num']).agg('mean')
    df['model'] = model
    df['dataset'] = constants.DATASET_NAMES[dataset]

    return df.reset_index()


def get_data_all(model, datasets):
    df = pd.concat([get_data(model, dataset) for dataset in datasets])
    return df


def plot_surprisal_vs_entropy(df_full, dataset):
    df['Surprisal'] = df['log_prob']
    df = pd.melt(df, id_vars=['new_ind', 'text_id', 'sentence_num', 'Surprisal'],
                 value_vars=['entropy', 'renyi_0.50'], var_name='Alpha',
                 value_name='Entropy')

    ENTROPY_NAMES = {
        'entropy': 'Shannon',
        'renyi_0.50': r'Rényi ($\alpha=.5$)',
    }
    df['Alpha'] = df['Alpha'].apply(lambda x: ENTROPY_NAMES[x])

    df = df.sample(n=4000, random_state=42)

    utils_plot.config_plots(width=3, height=2)
    fig = sns.lmplot(x='Surprisal', y='Entropy', hue='Alpha', row='dataset', data=df, legend=False)
    plt.legend()
    plt.xlim([0, 15])

    fname = 'results/plots/surprisal_vs_entropy--%s.pdf' % (dataset)
    fig.savefig(fname, bbox_inches='tight')


def subsample_dataframe(df, npoints=1000):
    dfs = []
    for dataset in df.dataset.unique():
        df_temp = df[df.dataset == dataset].copy()
        dfs += [df_temp.sample(n=npoints, random_state=42)]

    return pd.concat(dfs)


def plot_surprisal_vs_entropy_all(df):
    df = subsample_dataframe(df)

    df['Surprisal'] = df['log_prob']
    df = pd.melt(df, id_vars=['new_ind', 'text_id', 'sentence_num', 'dataset', 'Surprisal'],
                 value_vars=['entropy', 'renyi_0.50'], var_name='Alpha',
                 value_name='Entropy')

    ENTROPY_NAMES = {
        'entropy': 'Shannon',
        'renyi_0.50': r'Rényi ($\alpha=.5$)',
    }
    df['Alpha'] = df['Alpha'].apply(lambda x: ENTROPY_NAMES[x])
    df['Dataset'] = df['dataset']

    utils_plot.config_plots(width=6, height=2)
    fig = sns.lmplot(x='Surprisal', y='Entropy', hue='Alpha', col='Dataset', data=df, legend=False, aspect=(3/4)/1.3, height=3*1.3)
    plt.legend(ncol=2)
    plt.xlim([0, 15])
    plt.ylim([0, 25])
    fig.set_titles(col_template="{col_name}")

    fname = 'results/plots/surprisal_vs_entropy--all.pdf'
    fig.savefig(fname, bbox_inches='tight')


def plot_correlations(df_full):
    correlations = [['Dataset', 'Entropy', 'Correlation']]
    for dataset in df_full.dataset.unique():
        df = df_full[df_full.dataset == dataset].copy()

        corr, pvalue = scipy.stats.spearmanr(df['log_prob'], df['entropy'])
        correlations += [[df.dataset.unique()[0], 'Shannon', corr]]
        print(dataset, corr, pvalue)
        corr, pvalue = scipy.stats.spearmanr(df['log_prob'], df['renyi_0.50'])
        correlations += [[df.dataset.unique()[0], r'Rényi ($\alpha=.5$)', corr]]
        print(dataset, corr, pvalue)

    df = pd.DataFrame(correlations[1:], columns=correlations[0])

    utils_plot.config_plots(width=6, height=2)
    fig, ax = plt.subplots()
    sns.barplot(data=df, hue='Entropy', x='Dataset', y='Correlation')
    plt.legend(ncol=2, loc='lower center')
    plt.ylabel('')
    fig.savefig('results/plots/correlation--surprisal_vs_entropy.pdf', bbox_inches='tight')


def main():
    args = get_args()

    df = get_data_all(args.model, args.datasets)

    plot_correlations(df)
    plot_surprisal_vs_entropy_all(df)


if __name__ == '__main__':
    main()
