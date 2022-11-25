import sys
import os
import numpy as np
import pandas as pd
import scipy.special
import scipy.stats
import statsmodels.api as sm

import matplotlib as mpl
from matplotlib.ticker import FormatStrFormatter
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from utils import plot


# this function generates a perturbed version of p.
# a changes the temperature, b controls the size of the perturbation.
def perturb_distribution(logits, a, b):
  return scipy.special.softmax(a * logits + b * np.random.randn(logits.shape[0], logits.shape[1]), -1)


def get_renyi_entropy(p, alpha):
  if alpha == 1:
    return -np.sum(p * np.log(p), -1)
  else:
    return (1/(1-alpha)) * np.log(np.sum(p**alpha, -1))


# get the slope and p-value from a regression predicting RTs from surprisal and entropy of noisy model
def entropy_regression(reading_times, logits, a, b, alpha=1, is_baseline=False):
  q = perturb_distribution(logits, a, b)
  q_surprisal = -np.log(q[:, 0])

  if is_baseline:
    est = sm.OLS(reading_times, q_surprisal)
  else:
    q_entropy = get_renyi_entropy(q, alpha)
    est = sm.OLS(reading_times, np.column_stack((q_entropy, q_surprisal)))
  est2 = est.fit()
  llh = est2.llf

  return est2.params, est2.pvalues, llh


def analyse_renyi_predictor():
  # generate a random logits for categorical probability distribution p(y|x)
  dim = 100
  n = 2000
  logits = np.random.randn(n, dim)

  # Get true probability distribution
  p = scipy.special.softmax(logits, -1)
  assert (perturb_distribution(logits, 1, 0) == p).all()

  # Get reading times as surprisal of true distribution
  reading_times = -np.log(p[:, 0])

  # Define range of noise perturbations to analyse
  noise_range = [count / 25 for count in range(1, 26)]

  # Initialise temp variables usefull
  results = [['Contextual Entropy', 'Noise', 'Predictor', 'Slope', 'Slope $p$-value', r'$\Delta_{\mathtt{llh}}$']]

  # Get slopes and pvalues of Shannon entropy and surprisal when different noise scales are used.
  for noise_std in noise_range:
    for _ in range(10):
      _, _, llh_baseline = entropy_regression(reading_times, logits, 1, noise_std, is_baseline=True)
      params, pvalues, llh = entropy_regression(reading_times, logits, 1, noise_std, alpha=1.0)

      results += [['Shannon', noise_std, 'Entropy', params[0], pvalues[0], (llh - llh_baseline) / n]]
      results += [['Shannon', noise_std, 'Surprisal', params[1], pvalues[1], (llh - llh_baseline) / n]]

  # Get slopes and pvalues of Renyi entropy and surprisal when different noise scales are used.
  for noise_std in noise_range:
    for _ in range(10):
      # params, pvalues = eval_entropy(1, noise_std, alpha=0.5)
      _, _, llh_baseline = entropy_regression(reading_times, logits, 1, noise_std, is_baseline=True)
      params, pvalues, llh = entropy_regression(reading_times, logits, 1, noise_std, alpha=0.5)
      # print(f'\tStd:{noise_std:0.2f}. Params:', eval_entropy(1, noise_std, alpha=.5))
      results += [['Rényi', noise_std, 'Entropy', params[0], pvalues[0], (llh - llh_baseline) / n]]
      results += [['Rényi', noise_std, 'Surprisal', params[1], pvalues[1], (llh - llh_baseline) / n]]

  # Plot results
  plot.config_plots()
  df = pd.DataFrame(results[1:], columns=results[0])
  fig, ax = plt.subplots()
  sns.lineplot(data=df, x='Noise', y='Slope', hue='Contextual Entropy', style='Predictor', errorbar=('ci', 99))
  fig.savefig('results/renyi_analysis.pdf', bbox_inches='tight')
  plt.close()

  fig, ax = plt.subplots()
  df_plot = df[df['Predictor'] == 'Entropy'].copy()
  sns.lineplot(data=df_plot, x='Noise', y=r'$\Delta_{\mathtt{llh}}$', hue='Contextual Entropy', errorbar=('ci', 99))
  plt.plot(noise_range, np.zeros(25), color='C7', linestyle='dashed')
  fig.savefig('results/renyi_analysis--llh.pdf', bbox_inches='tight')
  plt.close()


def main():
  analyse_renyi_predictor()


if __name__ == '__main__':
    main()