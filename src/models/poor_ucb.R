# POOR UCB

ucb_poor <- function(K, N, mu, y_true, z_true, batch_size, 
                     burn = 1000, n_iter = 100, 
                     dynamics = c("common", "independent"),
                     model_path = NULL) {
  
  dynamics <- match.arg(dynamics)
  
  if (is.null(model_path)) {
    if (!exists("poor_model_path")) {
      stop("Model path not provided and poor_model_path is not defined.")
    }
    model_path <- poor_model_path
  }
  
  observed_rewards <- vector("list", K)
  selected_arms <- numeric(N)
  received_rewards <- numeric(N)
  regret <- numeric(N)
  
  # Oracle computation using ground-truth latent state 
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
  
  # Posterior tracking 
  posterior_matrices <- rep(list(1), K)
  z_last <- vector("list", K)
  a_t <- sample(1:K, 1)
  
  for (t in 1:N) {
    sampled_values <- numeric(K)
    
    # Parallel inference for all arms with enough data
    if (t %% batch_size == 0) {
      update_flags <- sapply(observed_rewards, function(x) length(x) >= 2)
      
      updated_posteriors <- future_lapply(1:K, function(l) {
        if (update_flags[l]) {
          data_list <- list(y = observed_rewards[[l]], N = length(observed_rewards[[l]]))
          model <- jags.model(model_path, data = data_list, n.chains = 1, quiet = TRUE)
          update(model, burn)
          post <- coda.samples(model, c("mu0", "mu1", "pi", paste0("z[", data_list$N, "]")), n.iter = n_iter)
          post_matrix <- as.matrix(post)
          list(post = post_matrix,
               z_last = post_matrix[, paste0("z[", data_list$N, "]")])
        } else {
          NULL
        }
      })
      
      for (l in 1:K) {
        if (!is.null(updated_posteriors[[l]])) {
          posterior_matrices[[l]] <- updated_posteriors[[l]]$post
          z_last[[l]] <- updated_posteriors[[l]]$z_last
        }
      }
    }
    
    # Arm selection via UCB logic
    for (l in 1:K) {
      if (!is.null(nrow(posterior_matrices[[l]])) && nrow(posterior_matrices[[l]]) > 0) {
        idx <- sample(1:nrow(posterior_matrices[[l]]), 1)
        mu0 <- posterior_matrices[[l]][idx, "mu0"]
        mu1 <- posterior_matrices[[l]][idx, "mu1"]
        
        pi0 <- posterior_matrices[[l]][, "pi[1]"]
        pi1 <- posterior_matrices[[l]][, "pi[2]"]
        sampled_pi <- ifelse(z_last[[l]] == 0, pi0, pi1)
        
        z_sampled <- rbinom(length(sampled_pi), size = 1, prob = sampled_pi)
        z_last[[l]] <- z_sampled
        
        sampled_mu <- ifelse(z_sampled == 0, mu0, mu1)
        sampled_values[l] <- mean(sampled_mu) + sd(sampled_mu) / sqrt(length(sampled_mu)) * log(t)
      } else {
        sampled_values[l] <- runif(1)
      }
    }
    
    a_t <- which.max(sampled_values)
    selected_arms[t] <- a_t
    r_t <- y_true[a_t, t]
    received_rewards[t] <- r_t
    
    true_z <- if (dynamics == "common") z_true[t] else z_true[a_t, t]
    regret[t] <- mu_selected[oracle_arm[t], t] - mu[a_t, true_z + 1]
    observed_rewards[[a_t]] <- c(observed_rewards[[a_t]], r_t)
  }
  
  return(list(
    cumulative_reward = cumsum(received_rewards),
    cumulative_regret = cumsum(regret),
    posterior_matrices = posterior_matrices
  ))
}
