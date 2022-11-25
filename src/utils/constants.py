import numpy as np

POWER_RANGE = np.arange(0., 3, 0.25)
STRIDE = 200
MODELS = ['gpt-small', 'gpt-medium', 'gpt-large', 'gpt-xl']

DATASETS = ['brown', 'natural_stories', 'provo_skip2zero', 'dundee_skip2zero', ]
ALL_DATASETS = ['brown', 'natural_stories',
                'provo', 'provo_skip2zero',
                'dundee', 'dundee_skip2zero',]

RENYI_RANGE = [.03, .06, .125, .25, .5, .75, 1.25, 1.5, 2, 5, 10]


PREDICTOR_NAMES = {
    'log_prob': r'$h(w_{t})$',
    'prev_log_prob': r'$h(w_{t\!-\!1})$',
    'prev2_log_prob': r'$h(w_{t\!-\!2})$',
    'prev3_log_prob': r'$h(w_{t\!-\!3})$',

    'entropy': r'$\mathrm{H}(W_{t})$',
    'prev_entropy': r'$\mathrm{H}(W_{t\!-\!1})$',
    'prev2_entropy': r'$\mathrm{H}(W_{t\!-\!2})$',
    'prev3_entropy': r'$\mathrm{H}(W_{t\!-\!3})$',
    'next_entropy': r'$\mathrm{H}(W_{t\!+\!1})$',

    'renyi_0.50': r'$\mathrm{H}_{.5}(W_{t})$',
    'prev_renyi_0.50': r'$\mathrm{H}_{.5}(W_{t\!-\!1})$',
    'prev2_renyi_0.50': r'$\mathrm{H}_{.5}(W_{t\!-\!2})$',
    'prev3_renyi_0.50': r'$\mathrm{H}_{.5}(W_{t\!-\!3})$',
    'next_renyi_0.50': r'$\mathrm{H}_{.5}(W_{t\!+\!1})$',

    'entropy_argmin': r'$\mathrm{H}_{\infty}(W_{t})$',
    'renyi_0.25': r'$\mathrm{H}_{.25}(W_{t})$',

    'prev_absdelta_entropy': r'$\left|h(w_{t\!-\!1}) - \mathrm{H}(W_{t\!-\!1})\right|$',
    'prev_absdelta_renyi_0.50': r'$\left|h(w_{t\!-\!1}) - \mathrm{H}_{.5}(W_{t\!-\!1})\right|$',
    'prev_overbudget_entropy': r'$\mathrm{ReLU}(\mathrm{H}(W_{t\!-\!1}) - h(w_{t\!-\!1}))$',
    'prev_overbudget_renyi_0.50': r'$\mathrm{ReLU}(\mathrm{H}_{.5}(W_{t\!-\!1}) - h(w_{t\!-\!1}))$',
}

PREDICTOR_ORDER = {
    'log_prob': 3,
    'prev_log_prob': 2,
    'prev2_log_prob': 1,
    'prev3_log_prob': 0,

    'entropy': 3,
    'prev_entropy': 2,
    'prev2_entropy': 1,
    'prev3_entropy': 0,
    'next_entropy': 4,

    'renyi_0.50': 3,
    'prev_renyi_0.50': 2,
    'prev2_renyi_0.50': 1,
    'prev3_renyi_0.50': 0,
    'next_renyi_0.50': 4,

    'prev_absdelta_entropy': 6,
    'prev_absdelta_renyi_0.50': 6,
    'prev_overbudget_entropy': 6,
    'prev_overbudget_renyi_0.50': 6,
}
PREDICTOR_ORDER_HIGHER = {
    'log_prob': 1,
    'prev_log_prob': 1,
    'prev2_log_prob': 1,
    'prev3_log_prob': 1,

    'entropy': 3,
    'prev_entropy': 3,
    'prev2_entropy': 3,
    'prev3_entropy': 3,
    'next_entropy': 3,

    'renyi_0.50': 2,
    'prev_renyi_0.50': 2,
    'prev2_renyi_0.50': 2,
    'prev3_renyi_0.50': 2,
    'next_renyi_0.50': 2,

    'prev_absdelta_entropy': 4,
    'prev_absdelta_renyi_0.50': 5,
    'prev_overbudget_entropy': 4,
    'prev_overbudget_renyi_0.50': 5,
}
DATASET_NAMES_FULL = {
    'natural_stories': 'Natural Stories',
    'brown': 'Brown',
    'provo': 'Provo (Progressive Gaze Duration, No Skipped)',
    'provo_skip2zero': 'Provo (Progressive Gaze Duration, Skipped Time=0)',
    'dundee': 'Dundee (Progressive Gaze Duration, No Skipped)',
    'dundee_skip2zero': 'Dundee (Progressive Gaze Duration, Skipped Time=0)',
}
DATASET_NAMES = {
    'natural_stories': 'Natural Stories',
    'brown': 'Brown',
    'provo': 'Provo',
    'provo_skip2zero': 'Provo',
    'dundee': 'Dundee',
    'dundee_skip2zero': 'Dundee',
}
DATASET_NAMES_PLOT = {
    'natural_stories': 'Natural Stories',
    'brown': 'Brown',
    'provo': 'Provo (No skip)',
    'provo_skip2zero': 'Provo (Skip=0)',
    'dundee': 'Dundee (No skip)',
    'dundee_skip2zero': 'Dundee (Skip=0)',
}
DATASET_ORDER = {
    'brown': 0,
    'natural_stories': 1,
    'provo': 4,
    'provo_skip2zero': 2,
    'dundee': 5,
    'dundee_skip2zero': 3,
}
