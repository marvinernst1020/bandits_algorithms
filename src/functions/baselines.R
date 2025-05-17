# BASELINE MODELS 

bandit_baselines <- function(algorithm, K, N, y_true, z_true, mu,
                             dynamics = c("independent", "common"),
                             batch_size = 1) {
  dynamics <- match.arg(dynamics)
  
  mu_selected <- matrix(0, nrow = K, ncol = N)
  if (dynamics == "independent") {
    for (i in 1:K) {
      for (t in 1:N) {
        mu_selected[i, t] <- mu[i, z_true[i, t] + 1]
      }
    }
  } else {
    for (i in 1:K) {
      for (t in 1:N) {
        mu_selected[i, t] <- mu[i, z_true[t] + 1]
      }
    }
  }
  oracle_arm <- apply(mu_selected, 2, which.max)
  
  counts <- rep(0, K)
  rewards <- rep(0, K)
  chosen_arm <- numeric(N)
  received_rewards <- numeric(N)
  cumulative_regret <- numeric(N)
  cumulative_reward <- numeric(N)
  
  warmup_order <- sample(1:K)
  regret <- 0
  total_reward <- 0
  current_arm <- NA
  
  for (t in 1:N) {
    if (t <= K) {
      idx <- warmup_order[t]
      current_arm <- idx  # set current_arm early
    } else if ((t - 1) %% batch_size == 0) {
      means <- ifelse(counts == 0, 0, rewards / counts)
      
      if (algorithm == "ucb") {
        ucb_values <- means + sqrt(2 * log(t) / ifelse(counts == 0, 1, counts))
        current_arm <- which.max(ucb_values)
        
      } else if (algorithm == "ucb-tuned") {
        variances <- pmin(1 / 4, means * (1 - means) + sqrt(2 * log(t) / ifelse(counts == 0, 1, counts)))
        ucb_tuned_values <- means + sqrt(log(t) / ifelse(counts == 0, 1, counts) * variances)
        current_arm <- which.max(ucb_tuned_values)
        
      } else if (algorithm == "ts") {
        samples <- rbeta(K, 1 + rewards, 1 + counts - rewards)
        current_arm <- which.max(samples)
        
      } else {
        stop("Unsupported algorithm: ", algorithm)
      }
      idx <- current_arm
    } else {
      idx <- current_arm  # this is now always defined
    }
    
    reward <- y_true[idx, t]
    counts[idx] <- counts[idx] + 1
    rewards[idx] <- rewards[idx] + reward
    total_reward <- total_reward + reward
    
    chosen_arm[t] <- idx
    received_rewards[t] <- reward
    
    regret_contrib <- if (dynamics == "independent") {
      mu_selected[oracle_arm[t], t] - mu[idx, z_true[idx, t] + 1]
    } else {
      mu_selected[oracle_arm[t], t] - mu[idx, z_true[t] + 1]
    }
    
    regret <- regret + regret_contrib
    cumulative_regret[t] <- regret
    cumulative_reward[t] <- total_reward
  }
  
  return(list(
    cumulative_regret = cumulative_regret,
    cumulative_reward = cumulative_reward,
    arm = chosen_arm
  ))
}


