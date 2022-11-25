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
    parser.add_argument('--predictor', type=str, default='entropy')

    return parser.parse_args()


def get_parameter_effects(model, datasets, glm_type, predictors, main_predictor):
    dfs = []

    budgeting_idx = 9 if main_predictor =='entropy' else 11
    budgeting_idx2 = 10 if main_predictor =='entropy' else 12
    DATASET2MODEL = {
        'natural_stories': ('prev_overbudget_%s' % main_predictor, budgeting_idx2),
        'brown': ('prev_overbudget_%s' % main_predictor, budgeting_idx),
        'provo': ('log_prob', 1),
        'provo_skip2zero': ('log_prob', 1),
        'dundee': (main_predictor, 5),
        'dundee_skip2zero': ('prev_overbudget_%s' % main_predictor, budgeting_idx),
    }

    fname_base = 'checkpoints/params/%s-%s-%s-predictor_%s-type_%d.tsv'
    for dataset in datasets:
        predictor_file, predictor_type = DATASET2MODEL[dataset]
        print(dataset)
        fname = fname_base % (glm_type, dataset, model, predictor_file, predictor_type)
        df_full = pd.read_csv(fname, sep='\t')
        for predictor in predictors:
            df = df_full.copy()
            df['model'] = model
            df['predictor'] = predictor
            df['dataset'] = dataset

            if predictor in df.columns:
                df['effect'] = df[predictor]
            else:
                df['effect'] = 0

            dfs += [df]

    df = pd.concat(dfs).reset_index()

    assert not df['effect'].isna().any()
    print(df.dataset.unique())

    return df


def main():
    args = get_args()
    args.datasets = args.datasets + ['provo', 'dundee']
    utils_plot.config_plots(height=5, width=8.3)

    predictor_times = ['']
    predictors = ['%s%s' % (predictor_time, args.predictor) for predictor_time in predictor_times]
    predictor_times = ['next_']
    predictors += ['%s%s' % (predictor_time, args.predictor) for predictor_time in predictor_times]
    predictor_times = ['', 'prev_', 'prev2_', 'prev3_']
    predictors += ['%s%s' % (predictor_time, 'log_prob') for predictor_time in predictor_times]
    predictors += ['prev_overbudget_%s' % args.predictor]

    df = get_parameter_effects(args.model, args.datasets, args.glm_type, predictors, args.predictor)
    print(df.dataset.unique())

    df['order2'] = df.predictor.apply(lambda x: constants.PREDICTOR_ORDER_HIGHER[x])
    df['order'] = df.predictor.apply(lambda x: constants.PREDICTOR_ORDER[x])
    df['order3'] = df.dataset.apply(lambda x: constants.DATASET_ORDER[x])
    df['word'] = df.predictor.apply(lambda x: constants.PREDICTOR_NAMES[x])
    df['dataset'] = df.dataset.apply(lambda x: constants.DATASET_NAMES_PLOT[x])
    df.sort_values(['order2', 'order', 'order3'], inplace=True)
    print(df.dataset.unique())

    fig, ax = plt.subplots()
    sns.barplot(hue='word', y='effect', x='dataset', data=df, errorbar=('ci', 68))

    plt.ylabel(r'Effect')
    plt.xlabel('')
    plt.legend(ncol=2)
    # ax.legend_.remove()
    plt.xlim([-.53, 5.5])
    ax.yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
    plt.ylim([-1.65, 6.75])
    plt.xticks(rotation=12)

    fname = 'results/plots/params--%s.pdf' % (args.predictor)
    fig.savefig(fname, bbox_inches='tight')


if __name__ == '__main__':
    main()
