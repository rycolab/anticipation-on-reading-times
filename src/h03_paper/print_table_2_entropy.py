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
    predictor_times = ['prev3_', 'prev2_', 'prev_', '']
    # predictors = [('entropy', 'replace'), ('entropy', 'add')]
    predictors = [(args.predictor, 'replace'), (args.predictor, 'add')]

    df_pvalue = get_pvalues(args.model, args.datasets, args.glm_type, predictors, predictor_times)
    print_deltallh(df_pvalue, args.datasets, predictors, predictor_times)


if __name__ == '__main__':
    main()
