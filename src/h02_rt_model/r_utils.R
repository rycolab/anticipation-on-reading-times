#!/usr/bin/env Rscript
shhh <- suppressPackageStartupMessages # It's a library, so shhh!

shhh(library( mgcv ))
shhh(library(dplyr))
shhh(library(ggplot2))
shhh(library(lme4))
# shhh(library( mgcv ))
theme_set(theme_bw())
shhh(library(tidymv))
shhh(library(gamlss))
shhh(library(gsubfn))
shhh(library(lmerTest))
shhh(library(hash))
options(digits=4)


load_data <- function(fname) {
  # Load data and handle slight pre-processing
  data <- read.csv(fname, header=T, sep='\t')
  
  return(data)
}


get_args <- function() {
  args <- commandArgs(trailingOnly = TRUE)
  argsLen <- length(args)

  input_fname <- args[1]
  output_fname <- args[2]
  params_output_fname_base <- args[3]
  
  merge_workers <- FALSE
  is_linear <- FALSE
  if (argsLen > 3) {
    if (args[4] == '--merge-workers') {
      merge_workers <- TRUE
    } else if (args[4] == '--is-linear') {
      is_linear <- TRUE
    }
  }
  if (argsLen > 4) {
    if (args[5] == '--merge-workers') {
      merge_workers <- TRUE
    } else if (args[5] == '--is-linear') {
      is_linear <- TRUE
    }
  }

  return(c(input_fname,output_fname,params_output_fname_base,merge_workers,is_linear))
}


get_logistic_llh <- function(form, trainData, testData, d_var, random_effects, is_linear){
  if(is_linear && random_effects) {
    sys.exit()
  } else if(is_linear && !random_effects)  {
    model <- glm(form,  data=trainData, family = "binomial")
  } else if(!is_linear && random_effects) {
    sys.exit()
  } else if(!is_linear && !random_effects)  {
    sys.exit()
  }

  probs <- predict(model, newdata=testData, type="response")
  estimate = ((testData[[d_var]] * log(probs)) +  ((1 - testData[[d_var]]) * log(1 - probs)))

  results <- list('estimate'=mean(estimate), 'model'=list(model))
  return(results)
}


get_regression_llh <- function(form, trainData, testData, d_var, random_effects, is_linear){
  if(is_linear && random_effects) {
    model <- lmer(form,  data=trainData, REML=FALSE)
  } else if(is_linear && !random_effects)  {
    model <- lm(form,  data=trainData)
  } else if(!is_linear && random_effects) {
    model <- gamr(form,  data=trainData, REML=FALSE)
  } else if(!is_linear && !random_effects)  {
    model <- gam(form,  data=trainData)
  }

  sigma <- mean(residuals(model)^2)
  estimate <- log(dnorm(testData[[d_var]],
                        mean=predict(model, newdata=testData, allow.new.levels=TRUE),
                        sd=sqrt(sigma)))

  results <- list('estimate'=mean(estimate), 'model'=list(model))
  return(results)
}


lme_cross_val <- function(form, df, d_var, num_folds=10, random_effects=TRUE, is_linear=TRUE, is_logistic=FALSE){
  folds <- cut(seq(1,nrow(df)), breaks=num_folds, labels=FALSE)

  estimates <- c()
  models <- c()
  for(i in 1:num_folds){
    testIndexes <- which(folds==i,arr.ind=TRUE)
    testData <- df[testIndexes,]
    trainData <- df[-testIndexes,]

    if(is_logistic) {
      list[estimate,model] <- get_logistic_llh(form, trainData, testData, d_var, random_effects, is_linear)
    } else {
      list[estimate,model] <- get_regression_llh(form, trainData, testData, d_var, random_effects, is_linear)
    }

    estimates <- c(estimates, estimate)
    models <- c(models, model)
  }

  results <- list('estimates'=estimates, 'models'=models)
  return(results)
}

relu <- function(df_column){
  new_value <- df_column
  # new_value[df_column > 0] <- df_column[df_column > 0]
  new_value[df_column < 0] <- 0

  return(new_value)
}

group_across_workers <- function(df_orig){
  # Convert from logical to numeric
  df_temp <- df_orig
  df_temp$outlier = (df_temp$outlier == "True")
  df_temp$outlier = as.numeric(df_temp$outlier)

  # Group Instances
  df = df_temp %>% group_by(new_ind,text_id,sentence_num) %>%
    summarise(time = mean(time),
              skipped = mean(skipped),

              log_prob = mean(log_prob),
              word_len = mean(word_len),
              freq = mean(freq),
              entropy = mean(entropy),
              entropy_argmin = mean(entropy_argmin),
              renyi_0.25 = mean(renyi_0.25),
              renyi_0.50 = mean(renyi_0.50),
              renyi_0.75 = mean(renyi_0.75),
              renyi_1.25 = mean(renyi_1.25),
              renyi_1.50 = mean(renyi_1.50),
              renyi_2.00 = mean(renyi_2.00),
              renyi_5.00 = mean(renyi_5.00),
              renyi_10.00 = mean(renyi_10.00),

              prev_log_prob = mean(prev_log_prob),
              prev_freq = mean(prev_freq),
              prev_word_len = mean(prev_word_len),
              prev_entropy = mean(prev_entropy),
              prev_entropy_argmin = mean(prev_entropy_argmin),
              prev_renyi_0.25 = mean(prev_renyi_0.25),
              prev_renyi_0.50 = mean(prev_renyi_0.50),
              prev_renyi_0.75 = mean(prev_renyi_0.75),
              prev_renyi_1.25 = mean(prev_renyi_1.25),
              prev_renyi_1.50 = mean(prev_renyi_1.50),
              prev_renyi_2.00 = mean(prev_renyi_2.00),
              prev_renyi_5.00 = mean(prev_renyi_5.00),
              prev_renyi_10.00 = mean(prev_renyi_10.00),

              prev2_log_prob = mean(prev2_log_prob),
              prev2_freq = mean(prev2_freq),
              prev2_word_len = mean(prev2_word_len),
              prev2_entropy = mean(prev2_entropy),
              prev2_entropy_argmin = mean(prev2_entropy_argmin),
              prev2_renyi_0.25 = mean(prev2_renyi_0.25),
              prev2_renyi_0.50 = mean(prev2_renyi_0.50),
              prev2_renyi_0.75 = mean(prev2_renyi_0.75),
              prev2_renyi_1.25 = mean(prev2_renyi_1.25),
              prev2_renyi_1.50 = mean(prev2_renyi_1.50),
              prev2_renyi_2.00 = mean(prev2_renyi_2.00),
              prev2_renyi_5.00 = mean(prev2_renyi_5.00),
              prev2_renyi_10.00 = mean(prev2_renyi_10.00),

              prev3_log_prob = mean(prev3_log_prob),
              prev3_freq = mean(prev3_freq),
              prev3_word_len = mean(prev3_word_len),
              prev3_entropy = mean(prev3_entropy),
              prev3_entropy_argmin = mean(prev3_entropy_argmin),
              prev3_renyi_0.25 = mean(prev3_renyi_0.25),
              prev3_renyi_0.50 = mean(prev3_renyi_0.50),
              prev3_renyi_0.75 = mean(prev3_renyi_0.75),
              prev3_renyi_1.25 = mean(prev3_renyi_1.25),
              prev3_renyi_1.50 = mean(prev3_renyi_1.50),
              prev3_renyi_2.00 = mean(prev3_renyi_2.00),
              prev3_renyi_5.00 = mean(prev3_renyi_5.00),
              prev3_renyi_10.00 = mean(prev3_renyi_10.00),

              next_log_prob = mean(next_log_prob),
              next_entropy = mean(next_entropy),
              next_entropy_argmin = mean(next_entropy_argmin),
              next_renyi_0.25 = mean(next_renyi_0.25),
              next_renyi_0.50 = mean(next_renyi_0.50),
              next_renyi_0.75 = mean(next_renyi_0.75),
              next_renyi_1.25 = mean(next_renyi_1.25),
              next_renyi_1.50 = mean(next_renyi_1.50),
              next_renyi_2.00 = mean(next_renyi_2.00),
              next_renyi_5.00 = mean(next_renyi_5.00),
              next_renyi_10.00 = mean(next_renyi_10.00),

              outlier = sum(outlier),

              .groups = 'drop')

  return(df)
}


get_predictor_budgeting_columns <- function(df, base_predictor){
  df[paste0('budget_delta_',base_predictor)] <- (df$log_prob - df[paste0(base_predictor)])
  df[paste0('overbudget_',base_predictor)] <- relu(- df[paste0('budget_delta_',base_predictor)])
  df[paste0('underbudget_',base_predictor)] <- relu(df[paste0('budget_delta_',base_predictor)])
  df[paste0('absdelta_',base_predictor)] <- abs(df[paste0('budget_delta_',base_predictor)])

  df[paste0('prev_budget_delta_',base_predictor)] <- (df$prev_log_prob - df[paste0('prev_',base_predictor)])
  df[paste0('prev_overbudget_',base_predictor)] <- relu(- df[paste0('prev_budget_delta_',base_predictor)])
  df[paste0('prev_underbudget_',base_predictor)] <- relu(df[paste0('prev_budget_delta_',base_predictor)])
  df[paste0('prev_absdelta_',base_predictor)] <- abs(df[paste0('prev_budget_delta_',base_predictor)])

  df[paste0('prev2_budget_delta_',base_predictor)] <- (df$prev2_log_prob - df[paste0('prev2_',base_predictor)])
  df[paste0('prev2_overbudget_',base_predictor)] <- relu(- df[paste0('prev2_budget_delta_',base_predictor)])
  df[paste0('prev2_underbudget_',base_predictor)] <- relu(df[paste0('prev2_budget_delta_',base_predictor)])
  df[paste0('prev2_absdelta_',base_predictor)] <- abs(df[paste0('prev2_budget_delta_',base_predictor)])

  df[paste0('prev3_budget_delta_',base_predictor)] <- (df$prev3_log_prob - df[paste0('prev3_',base_predictor)])
  df[paste0('prev3_overbudget_',base_predictor)] <- relu(- df[paste0('prev3_budget_delta_',base_predictor)])
  df[paste0('prev3_underbudget_',base_predictor)] <- relu(df[paste0('prev3_budget_delta_',base_predictor)])
  df[paste0('prev3_absdelta_',base_predictor)] <- abs(df[paste0('prev3_budget_delta_',base_predictor)])

  df[paste0('next_budget_delta_',base_predictor)] <- (df$next_log_prob - df[paste0('next_',base_predictor)])
  df[paste0('next_overbudget_',base_predictor)] <- relu(- df[paste0('next_budget_delta_',base_predictor)])
  df[paste0('next_underbudget_',base_predictor)] <- relu(df[paste0('next_budget_delta_',base_predictor)])
  df[paste0('next_absdelta_',base_predictor)] <- abs(df[paste0('next_budget_delta_',base_predictor)])

  return(df)
}


get_budgeting_columns <- function(df){
  df <- get_predictor_budgeting_columns(df, 'entropy')
  df <- get_predictor_budgeting_columns(df, 'renyi_0.50')

  return(df)
}


load_and_preprocess_data <- function(input_fname, merge_workers) {
  # Load data
  df <- load_data(input_fname)

  # Make skipped a numeric value
  if(! 'skipped' %in% colnames(df)) {
    df$skipped = FALSE
  }
  df$skipped = (df$skipped == "True")
  df$skipped = as.numeric(df$skipped)

  # Backup data
  df_orig <- df

  if(merge_workers) {
    # Merge results across workers
    df <- group_across_workers(df_orig)

    # Remove outliers
    df <- filter(df, df$outlier <= 2)
  } else {
    df <- df_orig
    # Subsample data for testing code
    df <- df[sample(nrow(df), 50000), ]

    # Remove outliers
    df <- filter(df, df$outlier == "False")
  }

  # Shuffle data
  shuffled_order <- sample(nrow(df))
  df <- df[shuffled_order,]

  # Define log_time
  df$log_time <- log(df$time)

  # Remove NaN
  df <- na.omit(df)

  # Get Budgeting columns
  df <- get_budgeting_columns(df)
}

get_baseline_score <- function(tgt_var, baseline_predictors, baseline_function, is_logistic=FALSE) {
    list[baseline_result,baseline_models] <- lme_cross_val(
      paste0(tgt_var," ~ ",baseline_predictors,baseline_function),
      df, tgt_var, random_effects=FALSE, is_logistic=is_logistic)

    return(baseline_result)
}
