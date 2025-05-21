## AR UCB 

ucb_ar <- function(K, N, mu, y_true, z_true, batch_size, 
                   burn = 1000, n_iter = 100,
                   dynamics = c("common","independent"),
                   model_path = NULL) {
  
  dynamics <- match.arg(dynamics)
  
  if (is.null(model_path)) {
    if (!exists("ar_model_path")) {
      stop("Model path not provided and ar_model_path is not defined.")
    }
    model_path <- ar_model_path
  }
  
  
  observed_rewards  <- vector("list", K)
  selected_arms     <- integer(N)
  received_rewards  <- numeric(N)
  regret            <- numeric(N)
  

  mu_selected <- matrix(0, nrow=K, ncol=N)
  if (dynamics=="common") {
    for (i in 1:K) for (t in 1:N)
      mu_selected[i,t] <- mu[i, z_true[t]+1]
  } else {
    for (i in 1:K) for (t in 1:N)
      mu_selected[i,t] <- mu[i, z_true[i,t]+1]
  }
  oracle_arm <- apply(mu_selected,2,which.max)
  
  posterior_matrices <- rep(list(NULL), K)
  
  for (t in 1:N) {
    if (t %% batch_size == 0) {
      res <- future.apply::future_lapply(seq_len(K), function(k) {
        yk <- observed_rewards[[k]]
        if (length(yk)<1) return(NULL)
        data_list <- list(y=yk, N=length(yk))
        m <- rjags::jags.model(model_path, data=data_list, n.chains=1, quiet=TRUE)
        update(m, burn)
        samp <- rjags::coda.samples(
          m,
          variable.names = paste0("logit_mu[", data_list$N, "]"),
          n.iter = n_iter
        )
        mat <- as.matrix(samp)
        mat[,1]  # just the column of logit_mu[N]
      }, future.seed=TRUE)
      posterior_matrices <- res
    }
    
    ucb_scores <- numeric(K)
    for (k in seq_len(K)) {
      logit_samps <- posterior_matrices[[k]]
      if (!is.null(logit_samps)) {
        p_samps <- 1/(1+exp(-logit_samps))        # invlogit
        n_k     <- length(p_samps)
        ucb_scores[k] <- mean(p_samps) +
          (sd(p_samps)/sqrt(n_k)) * log(t)
      } else {
        ucb_scores[k] <- Inf  
      }
    }
    
    a_t <- which.max(ucb_scores)
    selected_arms[t]    <- a_t
    r_t                 <- y_true[a_t, t]
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