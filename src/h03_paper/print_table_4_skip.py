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
from h03_paper.print_table_1_surprisal import get_pvalues, print_deltallh
from utils import constants
from utils import plot as utils_plot
from utils import utils


def get_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--datasets', type=str, nargs='+', default=['provo_skip2zero', 'dundee_skip2zero'])
    parser.add_argument('--model', type=str, default='gpt-small')
    parser.add_argument('--glm-type', type=str, default='skip-merged-linear')
    parser.add_argument('--predictor', type=str, default='entropy')
    # Other
    parser.add_argument('--seed', type=int, default=7)

    args = parser.parse_args()
    utils.config(args.seed)

    return args

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
    # elif pvalue < 0.05:
    #     pvalue_str = '$^{*}$\\phantom{$^{**}$}'
    elif pvalue < 0.05:
        pvalue_str = '$^{*}$\\phantom{$^{**}$}'
    else:
        pvalue_str = '\\phantom{$^{***}$}'

    return pvalue_str

def get_color_str(pvalue, is_positive_good, diff):
    if pvalue < .05:
        # if predictor_base != 'log_prob' and is_positive_good:
        if is_positive_good:
            color_diff = diff
        else:
            color_diff = - diff

        if color_diff > 0.0001:
            color_str = '\\textcolor{mygreen}{'
        elif color_diff < 0.0001:
            color_str = '\\textcolor{myred}{'
    else:
        color_str = '{'

    return color_str


def print_deltallh_skipped(df_pvalue, datasets, main_predictor, predictors, predictor_times, scaler=100):
    # for dataset in datasets:
    #     print_str = '%s ' % dataset_names[dataset]

    #     for predictor_base, predictor_type in predictors:

    print_strs = {
        ('log_prob', 'remove', 'provo_skip2zero'): r'\multirow{2}{*}{$[\surp]$ vs $[]$}',
        (main_predictor, 'add_scratch', 'provo_skip2zero'): r'\multirow{2}{*}{$[\ent]$ vs $[]$}',
        (main_predictor, 'replace', 'provo_skip2zero'): r'\multirow{2}{*}{$[\ent]$ vs $[\surp]$}',
        (main_predictor, 'add', 'provo_skip2zero'): r'\multirow{2}{*}{$[\ent; \surp]$ vs $[\surp]$}',
        ('log_prob', 'remove_from_%s' % main_predictor, 'provo_skip2zero'): r'\multirow{2}{*}{$[\ent; \surp]$ vs $[\ent]$}',
        (main_predictor, 'replace', 'provo'): r'\multirow{2}{*}{$[\ent]$ vs $[\surp]$}',
        (main_predictor, 'add', 'provo'): r'\multirow{2}{*}{$[\ent; \surp]$ vs $[\surp]$}',
    }

    for base_predictor, predictor_type in predictors:
        is_positive_good = True
        for dataset in datasets:
            if (base_predictor, predictor_type, dataset) in print_strs:
                print(print_strs[(base_predictor, predictor_type, dataset)])

            print_deltallh(
                df_pvalue, [dataset], [(base_predictor, predictor_type)], predictor_times,
                scaler=scaler, print_preffix='& ', is_positive_good=is_positive_good)
        print('[7pt]')
        print()



def main():
    args = get_args()
    predictor_times = ['prev3_', 'prev2_', 'prev_', '']
    predictors = [('log_prob', 'remove'), (args.predictor, 'add_scratch'), (args.predictor, 'replace'), (args.predictor, 'add'), ('log_prob', 'remove_from_%s' % args.predictor)]

    df_pvalue = get_pvalues(args.model, args.datasets, args.glm_type, predictors, predictor_times)
    print_deltallh_skipped(df_pvalue, args.datasets, args.predictor, predictors, predictor_times, scaler=1e4)


if __name__ == '__main__':
    main()
