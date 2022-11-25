import os
import sys
import argparse
import pandas as pd

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
    parser.add_argument('--glm-type', type=str, default='merged-linear')

    return parser.parse_args()


def get_entropy_llh(model, dataset, glm_type, use_full_predictor=False):
    dfs = []
    fname_base = 'checkpoints/delta_llh/%s-%s-%s.tsv'
    fname = fname_base % (glm_type, dataset, model)

    df = pd.read_csv(fname, sep='\t')
    df['model'] = model
    df['dataset'] = dataset
    df['dataset'] = df.dataset.apply(lambda x: constants.DATASET_NAMES[x])

    df_baseline = df[(df.predictor_type == 1) & (df.name == 'log_prob')].copy()
    df_baseline['type_name'] = '$h(w_t)$'
    df_baseline['type_order'] = 0
    df_baseline_expanded = []
    for alpha in [.25, .50, .75, 1.00, 1.25, 1.50, 2.00, 5.00, 10.00, float('inf')]:
        df_baseline_temp = df_baseline.copy()
        df_baseline_temp['alpha'] = alpha
        df_baseline_expanded += [df_baseline_temp]
    df_baseline_expanded = pd.concat(df_baseline_expanded)


    predictor_type_name = {
        5: r'$h(w_t) + \mathrm{H}_{\alpha}(w_t)$',
        6: r'$\mathrm{H}_{\alpha}(w_t)$',
    }
    predictor_order = {
        5: 2,
        6: 1,
    }
    df = df[df.name != 'log_prob']
    drop_keywords = ['prev', 'budget', 'delta', 'next']
    df = df[df.name.apply(lambda x: all([keyword not in x for keyword in drop_keywords]))]
    df = df[(df.predictor_type == 5) | (df.predictor_type == 6)]
    df['type_name'] = df.predictor_type.apply(lambda x: predictor_type_name[x])
    df['type_order'] = df.predictor_type.apply(lambda x: predictor_order[x])

    BASE_ENTROPIES_ALPHA = {
        'entropy': 1.0,
        'entropy_argmin': float('inf'),
    }
    df['alpha'] = df.name.apply(lambda x: float(x[6:]) if 'renyi_' in x else BASE_ENTROPIES_ALPHA[x])


    df = pd.concat([df, df_baseline_expanded]).reset_index()

    baseline_llh = df_baseline['diff_medium_logprob'].mean()
    return df, baseline_llh


def plot_dataset(dataset, args, use_full_predictor):
    df, baseline_llh  = get_entropy_llh(args.model, dataset, args.glm_type, use_full_predictor)

    alpha_str = {
        .25: '.25',
        .50: '.50',
        .75: '.75',
        1.00: '1.0',
        1.25: '1.2',
        1.50: '1.5',
        2.00: '2.0',
        5.00: '5.0',
        10.00: '10',
    }
    df['x_axis'] = df.alpha.apply(lambda x: alpha_str[x] if x < float('inf') else r'$\infty$')

    df.sort_values(['alpha', 'dataset', 'type_order'], inplace=True)

    df['diff'] = df['diff_medium_logprob'] * 100
    baseline_llh = baseline_llh * 100

    fig, ax = plt.subplots()
    sns.lineplot(x='x_axis', y='diff', hue='type_name', data=df, errorbar=('ci', 95), n_boot=20000)

    plt.ylabel(r'$\Delta_{\mathrm{llh}}$ ($10^{-2}$ nats)')
    plt.xlabel(r'$\alpha$')
    plt.xlim([0, 9])
    plt.xticks(rotation=30)
    plt.legend()
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))

    fname = 'results/plots/entropy--llh--%s.pdf' % (dataset)
    fig.savefig(fname, bbox_inches='tight')


def main():
    args = get_args()
    utils_plot.config_plots(width=3.5, height=6)

    for dataset in args.datasets:
        plot_dataset(dataset, args, use_full_predictor=False)


if __name__ == '__main__':
    main()
