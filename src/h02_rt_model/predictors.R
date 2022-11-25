
get_logprob_predictors <- function() {
  predictors <- c(
    # All, but one
    hash('name'='prev3_log_prob', 'type'=-3, 'function'='log_prob + prev_log_prob + prev2_log_prob'),
    hash('name'='prev2_log_prob', 'type'=-3, 'function'='log_prob + prev_log_prob + prev3_log_prob'),
    hash('name'='prev_log_prob', 'type'=-3, 'function'='log_prob + prev2_log_prob + prev3_log_prob'),
    hash('name'='log_prob', 'type'=-3, 'function'='prev_log_prob + prev2_log_prob + prev3_log_prob'),

    # Surprisals + entropy
    hash('name'='prev3_log_prob', 'type'=-4, 'function'='log_prob + entropy + prev_log_prob + prev2_log_prob'),
    hash('name'='prev2_log_prob', 'type'=-4, 'function'='log_prob + entropy + prev_log_prob + prev3_log_prob'),
    hash('name'='prev_log_prob', 'type'=-4, 'function'='log_prob + entropy + prev2_log_prob + prev3_log_prob'),
    hash('name'='log_prob', 'type'=-4, 'function'='entropy + prev_log_prob + prev2_log_prob + prev3_log_prob'),

    # Surprisals + renyi
    hash('name'='prev3_log_prob', 'type'=-5, 'function'='log_prob + renyi_0.50 + prev_log_prob + prev2_log_prob'),
    hash('name'='prev2_log_prob', 'type'=-5, 'function'='log_prob + renyi_0.50 + prev_log_prob + prev3_log_prob'),
    hash('name'='prev_log_prob', 'type'=-5, 'function'='log_prob + renyi_0.50 + prev2_log_prob + prev3_log_prob'),
    hash('name'='log_prob', 'type'=-5, 'function'='renyi_0.50 + prev_log_prob + prev2_log_prob + prev3_log_prob'),

    # All surprisals
    hash('name'='log_prob', 'type'=1, 'function'='log_prob+prev_log_prob+prev2_log_prob+prev3_log_prob')
  )
  predictors <- c(predictors, get_variable_predictors_next('log_prob'))

  return(predictors)
}


get_variable_predictors_prev <- function(predictor_base) {
    predictor <- paste0('prev_',predictor_base)

    predictors <- c(
        hash('name'=predictor, 'type'=5, 'function'=paste0('log_prob + prev_log_prob + ',predictor,' + prev2_log_prob + prev3_log_prob')),
        hash('name'=predictor, 'type'=6, 'function'=paste0('log_prob + ',predictor,' + prev2_log_prob + prev3_log_prob')),
        hash('name'=predictor, 'type'=9, 'function'=paste0('log_prob + entropy + prev_log_prob + ',predictor,' + prev2_log_prob + prev3_log_prob')),
        hash('name'=predictor, 'type'=11, 'function'=paste0('log_prob + renyi_0.50 + prev_log_prob + ',predictor,' + prev2_log_prob + prev3_log_prob'))
    )

    return(predictors)
}

get_variable_predictors_prev2 <- function(predictor_base) {
    predictor <- paste0('prev2_',predictor_base)

    predictors <- c(
        hash('name'=predictor, 'type'=5, 'function'=paste0('log_prob + prev_log_prob + prev2_log_prob + ',predictor,' + prev3_log_prob')),
        hash('name'=predictor, 'type'=6, 'function'=paste0('log_prob + prev_log_prob + ',predictor,' + prev3_log_prob')),
        hash('name'=predictor, 'type'=9, 'function'=paste0('log_prob + entropy + prev_log_prob + prev2_log_prob + ',predictor,' + prev3_log_prob')),
        hash('name'=predictor, 'type'=11, 'function'=paste0('log_prob + renyi_0.50 + prev_log_prob + prev2_log_prob + ',predictor,' + prev3_log_prob'))
    )

    return(predictors)
}

get_variable_predictors_prev3 <- function(predictor_base) {
    predictor <- paste0('prev3_',predictor_base)

    predictors <- c(
        hash('name'=predictor, 'type'=5, 'function'=paste0('log_prob + prev_log_prob + prev2_log_prob + prev3_log_prob + ',predictor)),
        hash('name'=predictor, 'type'=6, 'function'=paste0('log_prob + prev_log_prob + prev2_log_prob + ',predictor)),
        hash('name'=predictor, 'type'=9, 'function'=paste0('log_prob + entropy + prev_log_prob + prev2_log_prob + prev3_log_prob + ',predictor)),
        hash('name'=predictor, 'type'=11, 'function'=paste0('log_prob + renyi_0.50 + prev_log_prob + prev2_log_prob + prev3_log_prob + ',predictor))
    )

    return(predictors)
}

get_variable_predictors_next <- function(predictor_base) {
    predictor <- paste0('next_',predictor_base)

    predictors <- c(
        hash('name'=predictor, 'type'=5, 'function'=paste0('log_prob + prev_log_prob + prev2_log_prob + prev3_log_prob + ',predictor)),
        hash('name'=predictor, 'type'=9, 'function'=paste0('log_prob + entropy + ',predictor,' + prev_log_prob + prev2_log_prob + prev3_log_prob')),
        hash('name'=predictor, 'type'=11, 'function'=paste0('log_prob + renyi_0.50 + ',predictor,' + prev_log_prob + prev2_log_prob + prev3_log_prob'))
    )

    return(predictors)
}

get_variable_predictors_current <- function(predictor) {
    predictors <- c(
        hash('name'=predictor, 'type'=1, 'function'=paste0(predictor,' + prev_',predictor,' + prev2_',predictor,' + prev3_',predictor)),
        hash('name'=predictor, 'type'=5, 'function'=paste0('log_prob + ',predictor,' + prev_log_prob + prev2_log_prob + prev3_log_prob')),
        hash('name'=predictor, 'type'=6, 'function'=paste0(predictor,' + prev_log_prob + prev2_log_prob + prev3_log_prob')),
        hash('name'=predictor, 'type'=9, 'function'=paste0('log_prob + entropy + ',predictor,' + prev_log_prob + prev2_log_prob + prev3_log_prob')),
        hash('name'=predictor, 'type'=10, 'function'=paste0('log_prob + entropy + next_entropy + ',predictor,' + prev_log_prob + prev2_log_prob + prev3_log_prob')),
        hash('name'=predictor, 'type'=11, 'function'=paste0('log_prob + renyi_0.50 + ',predictor,' + prev_log_prob + prev2_log_prob + prev3_log_prob')),
        hash('name'=predictor, 'type'=12, 'function'=paste0('log_prob + renyi_0.50 + next_renyi_0.50 + ',predictor,' + prev_log_prob + prev2_log_prob + prev3_log_prob'))
    )

    return(predictors)
}

get_variable_predictors_all <- function(predictor) {
    predictors_current <- get_variable_predictors_current(predictor)
    predictors_prev <- get_variable_predictors_prev(predictor)
    predictors_prev2 <- get_variable_predictors_prev2(predictor)
    predictors_prev3 <- get_variable_predictors_prev3(predictor)
    predictors_next <- get_variable_predictors_next(predictor)

    predictors <- c(predictors_current, predictors_prev,
                    predictors_prev2, predictors_prev3,
                    predictors_next)
    return(predictors)
}
