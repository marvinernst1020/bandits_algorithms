#### AR with Thompson Sampling

thompson_ar <- function(K, N, mu, y_true, z_true, batch_size, 
                        burn = 1000, n_iter = 100,
                        dynamics = c("common", "independent"),
                        model_path = NULL) {
  
  dynamics <- match.arg(dynamics)
  
  if (is.null(model_path)) {
    if (!exists("ar_model_path")) {
      stop("Model path not provided and ar_model_path is not defined.")
    }
    model_path <- ar_model_path
  }
  
  # Oracle computation of best arm 
  observed_rewards <- vector("list", K)
  selected_arms <- numeric(N)
  received_rewards <- numeric(N)
  regret <- numeric(N)
  
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
  
  # Storage for posteriors
  posterior_matrices <- rep(list(NULL), K)
  
  for (t in 1:N) {
    if (t %% batch_size == 0) { #collected enough for a batch?
      res <- future.apply::future_lapply(1:K, function(k) {
        yk <- observed_rewards[[k]]
        if (length(yk) < 1) return(NULL)
        data_list <- list(y = yk, N = length(yk))
        m <- rjags::jags.model(model_path, data = data_list, n.chains = 1, quiet=TRUE)
        update(m, burn)
        # posterior of mu1 and last logit_mu[N]*
        samp <- coda::coda.samples(
          m,
          variable.names = c("mu1", ### p apo kjo?
                             paste0("logit_mu[", data_list$N, "]"),
                             "phi","sigma2"),
          n.iter = n_iter
        )
        mat <- as.matrix(samp)
        list(post = mat,
             last_logit = mat[, paste0("logit_mu[", data_list$N, "]")])
      }, future.seed=TRUE)
      for (k in seq_len(K)) posterior_matrices[[k]] <- res[[k]]
    }
    
    
    sampled_values <- numeric(K)
    for (k in seq_len(K)) {
      pm <- posterior_matrices[[k]]
      if (!is.null(pm)) {
        idx <- sample.int(nrow(pm$post), 1)
        logit_last <- pm$last_logit[idx]
        sampled_values[k] <- 1/(1+exp(-logit_last))  # invlogit
      } else {
        sampled_values[k] <- runif(1)
      }
    }
    
    # pick arm, observe reward, update
    a_t <- which.max(sampled_values)
    selected_arms[t] <- a_t
    r_t <- y_true[a_t, t]
    received_rewards[t] <- r_t
    observed_rewards[[a_t]] <- c(observed_rewards[[a_t]], r_t)
    regret[t] <- mu_selected[oracle_arm[t], t] - mu_selected[a_t, t]
  }
  
  list(
    cumulative_reward = cumsum(received_rewards),
    cumulative_regret = cumsum(regret),
    posterior_matrices = posterior_matrices
  )
}