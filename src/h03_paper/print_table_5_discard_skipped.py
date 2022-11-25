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
from h03_paper.print_table_4_skip import print_deltallh_skipped
from utils import constants
from utils import plot as utils_plot
from utils import utils


def get_args():
    parser = argparse.ArgumentParser()
    # Model
    parser.add_argument('--datasets', type=str, nargs='+', default=['provo', 'dundee'])
    parser.add_argument('--model', type=str, default='gpt-small')
    parser.add_argument('--glm-type', type=str, default='merged-linear')
    parser.add_argument('--predictor', type=str, default='entropy')
    # Other
    parser.add_argument('--seed', type=int, default=7)

    args = parser.parse_args()
    utils.config(args.seed)

    return args


def main():
    args = get_args()
    predictor_times = ['prev3_', 'prev2_', 'prev_', '']
    predictors = [(args.predictor, 'replace'), (args.predictor, 'add')]

    df_pvalue = get_pvalues(args.model, args.datasets, args.glm_type, predictors, predictor_times)
    print_deltallh_skipped(df_pvalue, args.datasets, args.predictor, predictors, predictor_times, scaler=100)


if __name__ == '__main__':
    main()
