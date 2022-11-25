import csv
import numpy as np
import mosestokenizer


MOSESTOKENIZER = mosestokenizer.MosesTokenizer("en")
MOSESDETOKENIZER = mosestokenizer.MosesDetokenizer("en")


class UnigramModel:
    # pylint: disable=too-few-public-methods

    def __init__(self, lang):
        self.lang = lang
        if lang == "en":
            with open('corpora/unigrams.csv', mode='r', encoding='utf8') as infile:
                reader = csv.reader(infile)
                self.lookup = {rows[0]:-float(rows[1]) for rows in reader}
        else:
            ValueError(f'Unknown language received {lang}')

    def __getitem__(self, key):
        if key == '':
            return np.nan

        try:
            tokens = [MOSESDETOKENIZER([t]) for t in MOSESTOKENIZER(key)]
            return np.sum([self.lookup.get(k, np.nan) for k in tokens])
        except KeyError:
            return np.nan


FREQ_MODEL = None
def frequency(word, lang="en"):
    # pylint: disable=global-statement
    global FREQ_MODEL
    if not FREQ_MODEL or FREQ_MODEL.lang != lang:
        FREQ_MODEL = UnigramModel(lang)
    return FREQ_MODEL[word]
