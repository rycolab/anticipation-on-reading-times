#!/usr/bin/env Rscript

# Load utils lib
source("src/h02_rt_model/r_utils.R")

# List of datasets to get statistics for
datasets = c('natural_stories', 'brown', 'provo_skip2zero', 'dundee_skip2zero', 'provo', 'dundee')
merge_workers = TRUE

# Loop through datasets
for (dataset in datasets) {
  input_fname <- paste0('checkpoints/rt_and_entropy/rt_vs_entropy-',dataset,'-gpt-small.tsv')

  set.seed(42)

  # Load data pre-processed and grouped by new_ind,text_id,sentence_num
  df_orig <- load_data(input_fname)
  print(paste0('Pre-processing dataset ',input_fname))
  df_final <- load_and_preprocess_data(input_fname, merge_workers)

  # Get statistics
  n_measurements = nrow(df_orig)
  n_workers = length(unique(df_orig$WorkerId))
  n_tokens = nrow(df_final)
  n_texts = length(unique(df_final$text_id))

  # Print for table
  print(paste0('    ',dataset,' & ',n_measurements,' & ',n_tokens,' & ',n_texts,' & ',n_workers,' \\'))
}
