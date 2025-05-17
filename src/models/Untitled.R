# ADAVANCED TS

thompson_advanced <- function(K, N, mu, y_true, z_true, batch_size, burn=1000, n_iter=100){
  
  model_adv <- "advanced_model.jags"
  observed_rewards <- list()
  selected_arms <- numeric(N)
  received_rewards <- numeric(N)
  regret <- numeric(N)
  
  # Oracle computation using ground-truth global latent state:
  mu_selected <- matrix(0, nrow = K, ncol = N)
  for (i in 1:K) {
    for (t in 1:N) {
      mu_selected[i, t] <- mu[i, z_true[t] + 1]
    }
  }
  oracle_arm <- apply(mu_selected, 2, which.max)
  
  # Storage for posterior tracking:
  post_matrix<- NULL
  
  # Initialization:
  selected_arms[1] <- sample(1:K, 1)
  observed_data <- matrix(NA, nrow = K, ncol = N)
  observed_data[selected_arms[1], 1] <- y_true[selected_arms[1], 1]
  received_rewards[1] <- y_true[selected_arms[1], 1]
  regret[1] <- mu_selected[oracle_arm[1], 1] - mu[selected_arms[1], z_true[1] + 1]
  
  # Start Thompson Sampling loop:
  for (t in 2:N) {
    sampled_values <- numeric(K)
    
    # Check if we have enough observed data:
    if (sum(!is.na(observed_data)) >= K * 2 && t%%Batch_size==0) {
      data_list <- list(
        y_obs = observed_data[, 1:(t-1)],
        K = K,
        N = t - 1
      )
      
      model <- jags.model(model_adv, data = data_list, n.chains = 1, quiet = TRUE)
      update(model, burn)
      post <- coda.samples(model, c("mu", "pi", paste0("z[", t - 1, "]")), n.iter = n_iter)
      post_matrix <- as.matrix(post)
    }
    
    if (is.matrix(post_matrix)) {
      idx <- sample(1:nrow(post_matrix), 1)
      
      pi0 <- post_matrix[idx,"pi[1]"]
      pi1 <- post_matrix[idx,"pi[2]"]
      
      if (t%%Batch_size==0) { z_last <- post_matrix[idx, paste0("z[", t-1, "]")]}
      
      sampled_pi <- ifelse(z_last == 0, pi0, pi1)
      z_t <- rbinom(length(sampled_pi), size = 1, prob = sampled_pi)
      z_last <- z_t
    }
    
    # Arm selection:
    for (i in 1:K) {
      if (is.matrix(post_matrix)) {
        mu0 <- post_matrix[idx, paste0("mu[", i, ",1]")]
        mu1 <- post_matrix[idx, paste0("mu[", i, ",2]")]
        sampled_values[i] <- (1 - z_t) * mu0 + z_t * mu1
      } else {
        sampled_values[i] <- runif(1)
      }
    }
    
    selected_arms[t] <- which.max(sampled_values)
    r_t <- y_true[selected_arms[t], t]
    received_rewards[t] <- r_t
    regret[t] <- mu_selected[oracle_arm[t], t] - mu[selected_arms[t], z_true[t] + 1]
    observed_data[selected_arms[t], t] <- r_t
  }
  
  cumulative_reward <- cumsum(received_rewards)
  cumulative_regret <- cumsum(regret)
  return(list(cumulative_reward=cumulative_reward,cumulative_regret=cumulative_regret,posterior_matrix=post_matrix))
}