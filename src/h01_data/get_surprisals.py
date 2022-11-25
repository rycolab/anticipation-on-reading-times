import os
import re
import sys
import argparse
import bisect
from string import punctuation

import numpy as np
import pandas as pd
import mosestokenizer

sys.path.insert(1, os.path.join(sys.path[0], '..'))
from corpus import process, metrics
from utils import constants, utils


def get_args():
    parser = argparse.ArgumentParser()
    # Data
    parser.add_argument('--input-path', type=str, required=True)
    parser.add_argument('--dataset', type=str, required=True)
    # Model
    parser.add_argument('--model', type=str, required=True, choices=constants.MODELS)
    # Output
    parser.add_argument('--output-fname', type=str, required=True)

    return parser.parse_args()


def process_natural_stories(args):
    gpt3_probs = pd.read_csv("%s/all_stories_gpt3.csv" % (args.input_path))
    # To get same indexing as stories db
    gpt3_probs["story"] = gpt3_probs["story"] + 1
    gpt3_probs['len'] = gpt3_probs.groupby("story", sort=False)['offset'].shift(periods=-1, fill_value=0) - gpt3_probs['offset']
    gpt3_probs['new_token'] = gpt3_probs.apply(lambda x: x['token'] if x['len'] == len(x['token']) else x['token'] + ' ', axis=1)

    stories_df = gpt3_probs.groupby(by=["story"], sort=False).agg({"new_token": [utils.string_join]}).reset_index()
    stories = list(zip(stories_df['story'], stories_df['new_token', 'string_join']))
    ns_stats = process.get_corpus_stats(stories, model=args.model)

    natural_stories = pd.read_csv("%s/processed_RTs.tsv" % (args.input_path), sep='\t').drop_duplicates()
    natural_stories.rename(columns={'RT': 'time',
                                    'item': 'text_id'}, inplace=True)
    natural_stories['new_ind'] = natural_stories['zone'] - 1
    natural_stories['sentence_num'] = natural_stories.apply(
        lambda x: bisect.bisect(ns_stats['sent_markers'][x['text_id']], x['new_ind']), axis=1)
    natural_stories = utils.find_outliers(natural_stories, transform=np.log)

    natural_stories = process.create_analysis_dfs(natural_stories, ns_stats, model=args.model)
    return natural_stories


def process_brown(args):
    brown = pd.read_csv('%s/brown_spr.csv' % (args.input_path))
    brown = brown.drop(columns='Unnamed: 0')
    brown.rename(columns = {'subject': 'WorkerId',
                            "text_pos":"Word_Number"}, inplace = True)
    # import ipdb; ipdb.set_trace()

    moses_normaliser = mosestokenizer.MosesPunctuationNormalizer("en")
    brown['word'] = brown.apply(lambda x: moses_normaliser(x['word']).strip(), axis=1)
    brown['word'] = brown.word.apply(lambda x: x.split()[0])

    inds, paragraphs = zip(*brown[['text_id','Word_Number','word']].drop_duplicates().dropna().groupby(by = ['text_id']).apply(lambda x: metrics.ordered_string_join(zip(x['Word_Number'], x['word']), ' ')))
    brown_stats = process.get_corpus_stats(list(enumerate(paragraphs)), model=args.model)

    brown['new_ind'] = brown.apply(lambda x: inds[x['text_id']].index(x["Word_Number"]), axis=1)
    brown['sentence_num'] = brown.apply(lambda x: bisect.bisect(brown_stats['sent_markers'][x['text_id']], x['new_ind']), axis=1)
    brown = utils.find_outliers(brown, transform=np.log)

    brown = process.create_analysis_dfs(brown, brown_stats, model=args.model)
    return brown


def process_provo(args, skip2zero=False, main_time_field='time'):
    provo = pd.read_csv('%s/provo.csv' % (args.input_path))
    provo.rename(columns = {'IA_DWELL_TIME':'time', 'Participant_ID': 'WorkerId', 'Word':'word',
                            "Text_ID":"text_id", "Sentence_Number":"sentence_num",
                           "IA_FIRST_RUN_DWELL_TIME": 'time2', 'IA_FIRST_FIXATION_DURATION':'time3'}, inplace = True)
    provo = provo.dropna(subset=["Word_Number"])
    provo = provo.astype({"Word_Number": 'Int64', "sentence_num": 'Int64'})

    # First Pass First Fixation Time
    provo['time4'] = provo['time3']
    provo.loc[provo['IA_SKIP'] == 1, 'time4'] = 0

    # First Pass Dwell Time
    provo['time5'] = provo['time2']
    provo.loc[provo['IA_SKIP'] == 1, 'time5'] = 0

    # Zero time if not fixated
    provo.loc[provo['time2'].isna(), 'time2'] = 0
    provo.loc[provo['time3'].isna(), 'time3'] = 0

    moses_normaliser = mosestokenizer.MosesPunctuationNormalizer("en")
    provo['word'] = provo.apply(lambda x: moses_normaliser(x['word']).strip(), axis=1)

    # fixing small discrepancy
    provo.loc[provo['word'] == '0.9', 'word'] = '90%'
    provo.loc[provo["word"] == 'women?s', 'word'] = 'womenõs'

    provo_text = pd.read_csv('%s/provo_norms.csv' % (args.input_path), encoding='latin-1')
    provo_text = provo_text[['Text_ID','Text']].drop_duplicates().sort_values(by=['Text_ID'])
    provo_text.drop(provo_text[(provo_text.Text_ID == 27) & (~provo_text.Text.str.contains("doesn't", regex=False))].index, inplace=True)
    inds = provo_text.apply(lambda x: list(range(1, len(x['Text'].split()) + 1)), axis=1)
    inds = {i: j for i, j in zip(provo_text['Text_ID'], inds)}
    paragraphs = {i: j.replace(u"\uFFFD", "?") for i, j in provo_text[['Text_ID', 'Text']].itertuples(index=False, name=None)}
    paragraphs_split = {i: [k.strip(punctuation) for k in j.lower().split()] for i, j in paragraphs.items()}

    provo_stats = process.get_corpus_stats(paragraphs.items(), model=args.model)

    provo["new_ind"] = provo["Word_Number"] - 2
    provo['new_ind'] = provo.apply(lambda x: x["new_ind"] + paragraphs_split[x['text_id']][x["new_ind"]:].index(x["word"].lower().strip(punctuation)), axis=1)
    provo['sentence_num'] = provo.apply(lambda x: bisect.bisect(provo_stats['sent_markers'][x['text_id']], x['new_ind']), axis=1)

    provo['time'] = provo[main_time_field]

    if not skip2zero:
        provo = utils.find_outliers(provo.loc[provo['time'] != 0].copy(), transform=np.log)
    else:
        provo = utils.find_outliers(provo, transform=np.log, ignore_zeros=True)
        provo['skipped'] = (provo['IA_SKIP'] == 1)

    provo = process.create_analysis_dfs(provo, provo_stats, model=args.model, dataset='provo')
    return provo


def mark_regressions(df):
    df = df.iloc[::-1].copy()

    df['fixation_number_temp'] = df['fixation_number']
    df.loc[df['fixation_number_temp'] == 0, 'fixation_number_temp'] = float('inf')
    df['future_min_fixation'] = df.groupby(['WorkerId', 'text_id'])['fixation_number_temp'].agg('cummin')

    df['is_regression'] = df['fixation_number'] > df['future_min_fixation']

    del df['fixation_number_temp']
    del df['future_min_fixation']

    return df.iloc[::-1]


def mark_first_pass(df):
    df_temp = df[df.fixation_number > 0].sort_values(['WorkerId', 'text_id', 'fixation_number'])
    df_temp['word_number_prev'] = df_temp.groupby(['WorkerId', 'text_id']).Word_Number.shift(1)
    df_temp.loc[df_temp['word_number_prev'].isna(), 'word_number_prev'] = 0

    df_temp['continued_pass'] = (df_temp['word_number_prev'] == df_temp['Word_Number'])
    df_temp['start_pass'] = ~(df_temp['continued_pass'])
    df_temp['pass_number'] = df_temp.groupby(['WorkerId', 'text_id']).start_pass.cumsum()

    df['pass_number'] = df_temp['pass_number']
    df.loc[df['pass_number'].isna(), 'pass_number'] = 0

    df['pass_number_min_temp'] = df.groupby(['WorkerId', 'text_id', 'Word_Number']).pass_number.transform('min')
    df['first_pass'] = (df.pass_number == df.pass_number_min_temp) | (df.pass_number.isna())

    del df['pass_number_min_temp']

    return df


def process_dundee(args, skip2zero=False, main_time_field='TotalReadingTime'):
    dundee_eyetracking_dir = '%s/eye-tracking' % (args.input_path)
    fileList = [
        os.path.join(dundee_eyetracking_dir, f)
        for f in os.listdir(dundee_eyetracking_dir) if re.match(r's\w\d+ma2p*\.dat', f)]
    cols = ['WorkerId', 'text_id', 'WORD','TEXT','LINE','OLEN','WLEN','XPOS','WNUM','FDUR','OBLP','WDLP','FXNO','TXFR']
    dundee = pd.DataFrame(columns = cols)
    for file in fileList:
        temp = pd.read_csv(file, sep='\s+', encoding='Windows-1252')
        match = re.search(r'(s\w)(\d+)ma2p*\.dat', file.split('/')[-1])
        subjId = match.group(1)
        text = int(match.group(2))
        temp.insert(loc=0, column='text_id', value=text)
        temp.insert(loc=0, column='WorkerId', value=subjId)
        dundee = dundee.append(temp)
    dundee.rename(columns = {'FDUR':'time', 'WORD':'word', 'WNUM': 'Word_Number', 'FXNO': 'fixation_number'}, inplace = True)
    dundee['time'] = dundee.time.astype('int64')
    dundee['fixation_number'] = dundee.fixation_number.astype('int64')
    dundee['Word_Number'] = dundee.Word_Number.astype('int64')
    dundee.reset_index(inplace=True)
    dundee = mark_regressions(dundee)
    dundee = mark_first_pass(dundee)
    dundee['is_progressive'] = dundee.first_pass & (~dundee.is_regression)
    dundee = dundee.reset_index().drop(columns=['index','OLEN','XPOS','OBLP','WDLP', 'TXFR'])

    dundee['TotalReadingTime'] = dundee.groupby(by=["WorkerId","text_id", "Word_Number"]).time.transform(np.nansum)
    dundee['FirstFixationTime'] = dundee.time
    dundee['FirstPassTime'] = dundee.groupby(by=["WorkerId","text_id", "Word_Number", 'pass_number']).time.transform(np.nansum)
    dundee['ProgressiveFirstFixationTime'] = dundee.FirstFixationTime * dundee.is_progressive
    dundee['ProgressiveFirstPassTime'] = dundee.FirstPassTime * dundee.is_progressive
    dundee.drop_duplicates(subset=["WorkerId","text_id", 'Word_Number'], inplace=True)

    # See Smith & Levy 2013
    dundee.drop(dundee.loc[dundee.WorkerId=='sg'].index, inplace=True)

    # Second block
    dundee_text_dir = '%s/texts' % (args.input_path)
    textList = [os.path.join(dundee_text_dir, f) for f in os.listdir(dundee_text_dir) if re.match(r'tx\d+wrdp\.dat', f)]
    cols = ['word', 'text_id', 'screen_nr', 'line_nr', 'pos_on_line', 'serial_nr', 'initial_letter_position', 'word_len_punct', 'word_len', 'punc_code', 'n_chars_before','n_chars_after', 'Word_Number', 'local_word_freq']
    dundeeTexts = pd.DataFrame(columns = cols)
    for text in textList:
        temp = pd.read_csv(text, sep='\s+', names=cols, encoding='Windows-1252')
        dundeeTexts = dundeeTexts.append(temp)
    moses_normaliser = mosestokenizer.MosesPunctuationNormalizer("en")
    dundee['word'] = dundee.apply(lambda x: re.sub(r"\s+", ' ', moses_normaliser(x['word'].strip().replace('""','"').replace('\n',' '))), axis=1)

    # Third block
    inds, paragraphs = zip(*dundeeTexts[['text_id','Word_Number','word']].drop_duplicates().dropna().groupby(by = ['text_id']).apply(lambda x: metrics.ordered_string_join(zip(x['Word_Number'], x['word']), ' ')))
    paragraphs = list(enumerate(paragraphs, 1))
    dundee_stats = process.get_corpus_stats(paragraphs, model=args.model)

    # Fourth block
    dundee = dundee.drop(dundee[dundee['word'].map(len) > 20].index)
    dundee['new_ind'] = dundee.apply(lambda x: inds[x['text_id']-1].index(x["Word_Number"]), axis=1)
    dundee['sentence_num'] = dundee.apply(lambda x: bisect.bisect(dundee_stats['sent_markers'][x['text_id']], x['new_ind']), axis=1)

    groupby_shape = dundee.groupby(by=["WorkerId","text_id", "sentence_num", "new_ind", 'word']).agg({'time': np.sum}).shape
    assert(groupby_shape[0] == dundee.shape[0])

    dundee['time'] = dundee[main_time_field]

    dundee = dundee.reset_index()

    if not skip2zero:
        dundee = utils.find_outliers(dundee.loc[dundee['time'] != 0].copy(), transform=np.log)
    else:
        dundee = utils.find_outliers(dundee, transform=np.log, ignore_zeros=True)
        dundee['skipped'] = (dundee['ProgressiveFirstFixationTime'] == 0)

    dundee = process.create_analysis_dfs(dundee, dundee_stats, model=args.model, dataset='dundee')
    return dundee


def process_dataset(args):
    if args.dataset == 'natural_stories':
        df = process_natural_stories(args)
    elif args.dataset == 'brown':
        df = process_brown(args)
    elif args.dataset == 'provo':
        df = process_provo(args, main_time_field='time5')
    elif args.dataset == 'provo_skip2zero':
        df = process_provo(args, main_time_field='time5', skip2zero=True)
    elif args.dataset == 'dundee':
        df = process_dundee(args, main_time_field='ProgressiveFirstPassTime')
    elif args.dataset == 'dundee_skip2zero':
        df = process_dundee(args, main_time_field='ProgressiveFirstPassTime', skip2zero=True)
    else:
        raise ValueError('Invalid dataset name: %s' % args.dataset)

    return df


def main():
    args = get_args()
    df = process_dataset(args)
    utils.write_tsv(df, args.output_fname)


if __name__ == '__main__':
    main()
