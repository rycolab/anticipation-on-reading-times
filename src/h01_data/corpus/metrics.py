# Helpers
from string import punctuation
import numpy as np
import nltk
from scipy.special import log_softmax, softmax
from intervaltree import Interval, IntervalTree

from models import unigram


def power(x, y):
    if x.mask.all():
        return np.nan
    return np.nanmean(x**y)


def ent(x):
    mod_x = np.nan_to_num(x.data, copy=True, nan=-np.inf)
    l_soft = log_softmax(-mod_x)
    return -np.sum(np.exp(l_soft)*l_soft)


def ent2(x):
    return np.sum(np.exp(-x)*x)


def r_ent(x, k=2):
    mod_x = np.nan_to_num(x.data, copy=True, nan=-np.inf)
    soft = softmax(-mod_x)
    return 1 / (1 - k) * np.log(np.sum(soft**k))


def r_ent2(x, k=2):
    return 1 / (1 - k) * np.log(np.sum(np.exp(-x)**k))


def local_diff(x):
    d = 0
    for i in range(len(x) - 1):
        d += abs(x[i + 1] - x[i])
    return d / len(x)


def local_diff2(x):
    d = 0
    for i in range(len(x) - 1):
        d += (x[i + 1] - x[i])**2
    return d / len(x)


def ordered_string_join(x, j=''):
    s = sorted(x, key=lambda x: x[0])
    a, b = list(zip(*s))
    return a, j.join(b)


def get_word_mapping(words):
    offsets = []
    pos = 0
    for w in words:
        offsets.append((pos, pos + len(w)))
        pos += len(w) + 1
    return offsets


def old_string_to_log_probs(string, probs, offsets):
    words = string.split()
    agg_log_probs = []
    word_mapping = get_word_mapping(words)
    cur_prob = 0
    cur_word_ind = 0
    last_ind = None
    for lp, ind in zip(probs, offsets):
        cur_prob += lp
        start, end = ind
        start_cur_word, end_cur_word = word_mapping[cur_word_ind]
        if end == end_cur_word:
            agg_log_probs.append(cur_prob)
            cur_prob = 0
            cur_word_ind += 1
        assert end <= end_cur_word
        last_ind = ind
    return agg_log_probs


def string_to_log_probs(string, probs, offsets):
    agg_surprisals = aggregate_string_results(string, probs, offsets, agg_func=agg_surprisal)
    return agg_surprisals


def string_to_entropies(string, probs, offsets):
    agg_entropies = aggregate_string_results(string, probs, offsets, agg_func=agg_entropy)
    return agg_entropies


def aggregate_string_results(string, probs, offsets, agg_func):
    words = string.split()
    word_mapping = get_word_mapping(words)

    t = IntervalTree()
    word_mapping = IntervalTree.from_tuples(
        [(x, y+1, idx) for idx, (x, y) in enumerate(word_mapping)])
    agg_results = {}

    for lp, ind in zip(probs, offsets):
        start, end = ind
        start = start + 1 if start != 0 else start
        interval = word_mapping[start:end+1]

        assert len(interval) == 1
        interval = sorted(interval)[0]

        idx = interval.data
        agg_func(agg_results, lp, idx)

    return [agg_results[i] for i in range(len(words))]


def agg_surprisal(agg_results, lp, idx):
    agg_results[idx] = agg_results.get(idx, 0) + lp


def agg_entropy(agg_results, lp, idx):
    if idx not in agg_results:
        agg_results[idx] = lp


def old_string_to_entropies(string, probs, offsets):
    words = string.split()
    agg_log_probs = []
    word_mapping = get_word_mapping(words)
    cur_prob = 0
    cur_word_ind = 0
    last_ind = None
    for lp, ind in zip(probs, offsets):
        cur_prob += lp
        start, end = ind
        start = start + 1 if start != 0 else start
        start_cur_word, end_cur_word = word_mapping[cur_word_ind]
        if start == start_cur_word:
            agg_log_probs.append(cur_prob)

        if end == end_cur_word:
            cur_prob = 0
            cur_word_ind += 1

        if start < start_cur_word:
            import ipdb; ipdb.set_trace()
        assert start >= start_cur_word
        assert end <= end_cur_word
        last_ind = ind
    return agg_log_probs


def string_to_uni_log_probs(string):
    words = [s.strip().strip(punctuation).lower() for s in string.split()]
    return [unigram.frequency(w.strip().strip(punctuation)) for w in words]
