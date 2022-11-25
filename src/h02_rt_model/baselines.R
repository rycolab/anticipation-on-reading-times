get_baselines <- function() {
  baselines <- c(
    hash('name'='medium_logprob', 'function'='+prev_log_prob+prev2_log_prob+prev3_log_prob'),
    hash('name'='full_logprob', 'function'='+log_prob+prev_log_prob+prev2_log_prob+prev3_log_prob'),

    hash('name'='successor_entropy_logprob', 'function'='+ log_prob + next_entropy + prev_log_prob + prev2_log_prob + prev3_log_prob'),
    hash('name'='both_entropy', 'function'='+ log_prob + entropy + prev_log_prob + prev2_log_prob + prev3_log_prob'),

    hash('name'='successor_renyi_logprob', 'function'='+ log_prob + next_renyi_0.50 + prev_log_prob + prev2_log_prob + prev3_log_prob'),
    hash('name'='both_renyi', 'function'='+ log_prob + renyi_0.50 + prev_log_prob + prev2_log_prob + prev3_log_prob')
  )

  return(baselines)
}