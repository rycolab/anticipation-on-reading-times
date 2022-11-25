import os
import sys
import argparse
import pandas as pd
import numpy as np

import matplotlib as mpl
# import matplotlib.font_manager
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import constants
from utils import utils


def get_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--datasets', type=str, nargs='+', default=constants.DATASETS)
    parser.add_argument('--model', type=str, default='gpt-small')
    parser.add_argument('--glm-type', type=str, default='merged-linear')
    parser.add_argument('--predictor', type=str, default='entropy')
    # Other
    parser.add_argument('--seed', type=int, default=7)

    args = parser.parse_args()
    utils.config(args.seed)

    return args


def get_spillover_llh(model, datasets, glm_type, predictor, predictor_type, predictor_times):
    dfs = []
    fname_base = 'checkpoints/delta_llh/%s-%s-%s.tsv'
    for dataset in datasets:
        fname = fname_base % (glm_type, dataset, model)
        df = pd.read_csv(fname, sep='\t')
        df['model'] = model
        df['dataset'] = dataset

        predictor_type_dict = {
            'remove': -3,
            'remove_from_entropy': -4,
            'remove_from_renyi_0.50': -5,
            'add': 5,
            'add_scratch': 6,
            'replace': 6,
            'budget_entropy': 9,
            'successor_entropy': 10,
            'budget_renyi_0.50': 11,
            'successor_renyi_0.50': 12,
        }

        df = df[df.predictor_type == predictor_type_dict[predictor_type]]

        if predictor == 'log_prob' and predictor_type == 'remove':
            df['diff'] = - df['diff_full_logprob']
        elif predictor in ['entropy', 'renyi_0.50'] and predictor_type in ['add', 'replace']:
            df['diff'] = df['diff_full_logprob']
        elif predictor in ['entropy', 'renyi_0.50'] and predictor_type in ['add_scratch', 'replace']:
            df['diff'] = df['diff_medium_logprob']
        elif predictor in ['entropy'] and predictor_type in ['successor_entropy']:
            df['diff'] = df['diff_successor_entropy_logprob']
        elif predictor in ['renyi_0.50'] and predictor_type in ['successor_renyi_0.50']:
            df['diff'] = df['diff_successor_renyi_logprob']
        elif predictor in ['next_entropy', 'next_log_prob', 'next_renyi_0.50'] and predictor_type in ['add']:
            df['diff'] = df['diff_full_logprob']
        elif predictor in ['next_entropy'] and predictor_type in ['budget_entropy']:
            df['diff'] = df['diff_both_entropy']
        elif predictor in ['next_renyi_0.50'] and predictor_type in ['budget_renyi_0.50']:
            df['diff'] = df['diff_both_renyi']
        elif predictor == 'log_prob' and predictor_type == 'remove_from_entropy':
            df['diff'] = - df['diff_both_entropy']
        elif predictor == 'log_prob' and predictor_type == 'remove_from_renyi_0.50':
            df['diff'] = - df['diff_both_renyi']
        elif predictor in ['entropy', 'overbudget_entropy', 'underbudget_entropy', 'absdelta_entropy'] and predictor_type == 'budget_entropy':
            df['diff'] = df['diff_both_entropy']
        elif predictor in ['renyi_0.50', 'overbudget_renyi_0.50', 'underbudget_renyi_0.50', 'absdelta_renyi_0.50'] and predictor_type == 'budget_renyi_0.50':
            df['diff'] = df['diff_both_renyi']
        else:
            raise ValueError('Wrong predictor--type combination (%s, %s)' % (predictor, predictor_type))

        dfs += [df]

    df = pd.concat(dfs).reset_index()

    predictors = ['%s%s' % (predictor_time, predictor) for predictor_time in predictor_times]
    df = df[df.name.isin(predictors)]

    return df


def permutation_test(array, n_permuts=500000):
    mean = abs(array.mean())

    permuts = np.random.randint(0, 2, size=(n_permuts, array.shape[0])) * 2 - 1

    permut_means = np.abs((array * permuts).mean(-1))
    n_larger = (permut_means > mean).sum()

    return (n_larger + 1) / n_permuts


def get_sign_str(diff):
    if diff >= 0:
        sign_str = '\\phantom{-}'
    else:
        sign_str = ''

    return sign_str


def get_pvalue_str(pvalue):
    if pvalue < 0.001:
        pvalue_str = '$^{***}$'
    elif pvalue < 0.01:
        pvalue_str = '$^{**}$\\phantom{$^{*}$}'
    elif pvalue < 0.05:
        pvalue_str = '$^{*}$\\phantom{$^{**}$}'
    else:
        pvalue_str = '\\phantom{$^{***}$}'

    return pvalue_str


def get_color_str(pvalue, is_positive_good, diff):
    color_str = '{'

    if pvalue < .05:
        if is_positive_good:
            color_diff = diff
        else:
            color_diff = - diff

        if color_diff > 0.0001:
            color_str = '\\textcolor{mygreen}{'
        elif color_diff < 0.0001:
            color_str = '\\textcolor{myred}{'

    return color_str


def print_deltallh(df, datasets, predictors, predictor_times, scaler=100,
                   print_preffix='', is_positive_good=True,
                   dataset_names=constants.DATASET_NAMES):
    for dataset in datasets:
        print_str = '%s%s ' % (print_preffix, dataset_names[dataset])

        for predictor_base, predictor_type in predictors:
            for predictor_time in predictor_times:
                predictor = '%s%s' % (predictor_time, predictor_base)

                df_temp = df[(df.dataset == dataset) &
                             (df.predictor == predictor) &
                             (df.predictor_type == predictor_type)]
                if df_temp.shape[0] == 0:
                    continue

                diff = df_temp.diffs.item()
                pvalue = df_temp['pvalue-corrected'].item()

                sign_str = get_sign_str(diff)
                pvalue_str = get_pvalue_str(pvalue)
                color_str = get_color_str(pvalue, is_positive_good, diff)

                print_str += '& %s%s%.2f}%s' % (sign_str, color_str, diff * scaler, pvalue_str)
        print_str += ' \\\\'

        print(print_str)


def get_corrections(df):
    for _, row in df[['predictor', 'predictor_type']].drop_duplicates().iterrows():
        predictor, predictor_type = row.predictor, row.predictor_type
        df_temp = df[(df.predictor == predictor) & (df.predictor_type == predictor_type)].copy()
        df_temp.sort_values('pvalue', inplace=True)
        df_temp['rank'] = range(df_temp.shape[0])
        df_temp['pvalue-corrected'] = df_temp['pvalue'] * df_temp.shape[0] / (df_temp['rank'] + 1)

        df.loc[(df.predictor == predictor) & (df.predictor_type == predictor_type), 'pvalue-corrected'] = df_temp['pvalue-corrected']

    return df


def get_pvalues(model, datasets, glm_type, predictors, predictor_times):
    pvalues = []
    for base_predictor, predictor_type in predictors:
        df = get_spillover_llh(model, datasets, glm_type, base_predictor, predictor_type, predictor_times)

        for dataset in df.dataset.unique():
            for predictor in df.name.unique():
                df_temp = df[(df.dataset == dataset) & (df.name == predictor)]

                diffs = df_temp['diff'].to_numpy()
                p_value = permutation_test(diffs)

                pvalues += [[dataset, predictor, predictor_type, diffs.mean(), p_value]]
                # print('Delta llh for %s in %s is: \t%.4f. p_value is: \t%.4f' % (predictor, dataset, diffs.mean() * 100, p_value))

    df_pvalue = pd.DataFrame(pvalues, columns=['dataset', 'predictor', 'predictor_type', 'diffs', 'pvalue'])
    df_pvalue = get_corrections(df_pvalue)
    return df_pvalue


def main():
    args = get_args()
    predictor_times = ['prev3_', 'prev2_', 'prev_', '']
    predictors = [('log_prob', 'remove')]

    df_pvalue = get_pvalues(args.model, args.datasets, args.glm_type, predictors, predictor_times)
    print_deltallh(df_pvalue, args.datasets, predictors, predictor_times)


if __name__ == '__main__':
    main()
