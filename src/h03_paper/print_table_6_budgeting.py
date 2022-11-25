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
from h03_paper.print_table_1_surprisal import get_args, get_pvalues, print_deltallh
from utils import constants
from utils import plot as utils_plot


def main():
    args = get_args()
    args.datasets = args.datasets + ['provo', 'dundee']
    predictor_times = ['prev3_', 'prev2_', 'prev_', ]
    predictors = [('%s%s' % (x, args.predictor), 'budget_%s' % args.predictor)
                  for x in ['', 'overbudget_', 'underbudget_', 'absdelta_']]

    df_pvalue = get_pvalues(args.model, args.datasets, args.glm_type, predictors, predictor_times)
    print_deltallh(df_pvalue, args.datasets, predictors, predictor_times)


if __name__ == '__main__':
    main()
