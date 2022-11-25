import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns


def config_plots(height=4, width=10, font_scale=1.5, verbose=False):
    fig_size = plt.rcParams["figure.figsize"]
    fig_size[0] = width
    fig_size[1] = height

    sns.set_palette("Set2")
    sns.set_context("notebook", font_scale=font_scale)
    mpl.rc('font', family='serif', serif='Times New Roman')

    params = {'text.usetex': False, 'mathtext.fontset': 'stix'}
    plt.rcParams.update(params)

    if verbose:
        print(plt.rcParams['axes.prop_cycle'].by_key()['color'])
