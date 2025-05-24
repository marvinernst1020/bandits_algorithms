
##  AR-UCB-KFAS


ucb_ar_kfas <- function(K, N, mu, y_true, z_true, batch_size,
                        burn = 1000, n_iter = 100,
                        dynamics = c("common", "independent"),
                        model_path = NULL) {
  
  dynamics <- match.arg(dynamics)
  if (dynamics == "independent") {
    if (!is.matrix(z_true))
      stop("For 'independent' dynamics z_true must be a matrix [K x N] or [N x K].")
    if (nrow(z_true) == N && ncol(z_true) == K)
      z_true <- t(z_true)                                   # auto-transpose
    if (nrow(z_true) != K || ncol(z_true) != N)
      stop(glue::glue("z_true dims should be [K x N]; got [{nrow(z_true)} x {ncol(z_true)}]"))
  } else {                                                  # common
    if (!is.vector(z_true) || length(z_true) != N)
      stop(glue::glue("For 'common' dynamics z_true must be length N = {N}."))
  }

  observed_rewards  <- vector("list", K)
  selected_arms     <- integer(N)
  received_rewards  <- numeric(N)
  regret            <- numeric(N)
  smoothed_models   <- rep(list(NULL), K)                   
  
  mu_selected <- matrix(0, K, N)
  for (i in 1:K)
    for (t in 1:N)
      mu_selected[i, t] <- if (dynamics == "independent")
        mu[i, z_true[i, t] + 1] else
          mu[i, z_true[t] + 1]
  oracle_arm <- as.integer(apply(mu_selected, 2, which.max))
  for (t in 1:N) {
    if (t %% batch_size == 0) {
      for (k in 1:K) {
        y_k <- observed_rewards[[k]]
        if (length(y_k) >= 2)                       
          smoothed_models[[k]] <- fit_kfas_bernoulli_ar(y_k)
      }
    }
    
    ucb_scores <- numeric(K)
    
    for (k in 1:K) {
      mod <- smoothed_models[[k]]
      
      if (!is.null(mod)) {
        logit_mu_last <- tail(mod$smoothed_logit_mu, 1)
        sd_logit <- 0.1

        draws      <- rnorm(n_iter, logit_mu_last, sd_logit)
        p_draws    <- inv_logit(draws)
        mean_p     <- mean(p_draws)
        se_p       <- sd(p_draws) / sqrt(length(p_draws))
        
        ucb_scores[k] <- mean_p + se_p * log(t)
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
    smoothed_models   = smoothed_models
  )
}