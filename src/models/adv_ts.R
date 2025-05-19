# ADAVANCED TS

thompson_advanced <- function(K, N, mu, y_true, z_true, batch_size,
                              burn = 1000, n_iter = 100,
                              window_size = Inf,
                              dynamics = c("common", "independent"),
                              model_path = NULL) {
  
  dynamics <- match.arg(dynamics)
  
  if (is.null(model_path)) {
    if (!exists("advanced_model_path")) {
      stop("Model path not provided and advanced_model_path is not defined.")
    }
    model_path <- advanced_model_path
  }
  
  observed_rewards <- list()
  selected_arms <- numeric(N)
  received_rewards <- numeric(N)
  regret <- numeric(N)
  post_matrix <- NULL
  
  # Oracle computation
  mu_selected <- matrix(0, nrow = K, ncol = N)
  if (dynamics == "common") {
    for (i in 1:K) {
      for (t in 1:N) {
        mu_selected[i, t] <- mu[i, z_true[t] + 1]
      }
    }
  } else {
    for (i in 1:K) {
      for (t in 1:N) {
        mu_selected[i, t] <- mu[i, z_true[i, t] + 1]
      }
    }
  }
  oracle_arm <- apply(mu_selected, 2, which.max)
  
  # Initialization
  selected_arms[1] <- sample(1:K, 1)
  observed_data <- matrix(NA, nrow = K, ncol = N)
  observed_data[selected_arms[1], 1] <- y_true[selected_arms[1], 1]
  received_rewards[1] <- y_true[selected_arms[1], 1]
  
  regret[1] <- mu_selected[oracle_arm[1], 1] - 
    if (dynamics == "common") mu[selected_arms[1], z_true[1] + 1] else mu[selected_arms[1], z_true[selected_arms[1], 1] + 1]
  
  # TS loop
  for (t in 2:N) {
    sampled_values <- numeric(K)
    
    if (sum(!is.na(observed_data)) >= K * 2 && t %% batch_size == 0) {
      # sliding window logic
      start_idx <- max(1, t - window_size)
      y_window <- observed_data[, start_idx:(t - 1)]
      N_window <- ncol(y_window)
      
      data_list <- list(
        y_obs = y_window,
        K = K,
        N = N_window
      )
      
      model <- jags.model(model_path, data = data_list, n.chains = 1, quiet = TRUE)
      update(model, burn)
      post <- coda.samples(model, c("mu", "pi", paste0("z[", N_window, "]")), n.iter = n_iter)
      post_matrix <- as.matrix(post)
      ## avoid label‐switching:
      post_matrix <- t(apply(post_matrix, 1, function(draw) {
        if (draw["mu[1,1]"] > draw["mu[1,2]"]) {
          tmp11 <- draw["mu[1,1]"]; tmp21 <- draw["mu[2,1]"]
          draw["mu[1,1]"] <- draw["mu[1,2]"]
          draw["mu[2,1]"] <- draw["mu[2,2]"]
          draw["mu[1,2]"] <- tmp11
          draw["mu[2,2]"] <- tmp21
          tmp_pi <- draw["pi[1]"]
          draw["pi[1]"] <- draw["pi[2]"]
          draw["pi[2]"] <- tmp_pi
        }
        draw
      })) 
      message(glue::glue("[{Sys.time()}] Performed inference at time step {t} (window {start_idx}:{t-1})"))
    }
    
    if (is.matrix(post_matrix)) {
      idx <- sample(1:nrow(post_matrix), 1)
      pi0 <- post_matrix[idx, "pi[1]"]
      pi1 <- post_matrix[idx, "pi[2]"]
      if (t %% batch_size == 0) {
        z_last <- post_matrix[idx, paste0("z[", N_window, "]")]
      }
      sampled_pi <- ifelse(z_last == 0, pi0, pi1)
      z_t <- rbinom(1, 1, prob = sampled_pi)
      z_last <- z_t
    }
    
    # Arm selection
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
    observed_data[selected_arms[t], t] <- r_t
    
    true_z <- if (dynamics == "common") z_true[t] else z_true[selected_arms[t], t]
    regret[t] <- mu_selected[oracle_arm[t], t] - mu[selected_arms[t], true_z + 1]
  }
  
  return(list(
    cumulative_reward = cumsum(received_rewards),
    cumulative_regret = cumsum(regret),
    posterior_matrix = post_matrix
  ))
}
