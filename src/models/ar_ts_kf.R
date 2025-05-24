##############################################################################
##  Thompson Sampling with KFAS AR(1)  –  legacy-compatible wrapper
##  * accepts burn, n_iter, ... so existing call-sites don’t break
##############################################################################

thompson_ar_kfas <- function(K, N, mu, y_true, z_true,
                             batch_size = 20,
                             burn       = NULL,  # <- kept for compatibility
                             n_iter     = NULL,  # <-
                             dynamics   = c("common", "independent"),
                             ...) {             # swallow any extra args
  dynamics <- match.arg(dynamics)
  
  if (!is.null(burn) || !is.null(n_iter))
    message("[thompson_ar_kfas] 'burn' and 'n_iter' are ignored in the ",
            "KFAS implementation (kept only for API compatibility).")
  
  ## ----- latent-state checks (unchanged) ---------------------------------
  if (dynamics == "independent") {
    if (!is.matrix(z_true))
      stop("For 'independent' dynamics, z_true must be a K×N matrix.")
    if (nrow(z_true) == N && ncol(z_true) == K) z_true <- t(z_true)
    if (nrow(z_true) != K || ncol(z_true) != N)
      stop(sprintf("z_true must be %d×%d; got %d×%d",
                   K, N, nrow(z_true), ncol(z_true)))
  } else {
    if (length(z_true) != N)
      stop("For 'common' dynamics, z_true must be a vector of length N.")
  }
  
  ## ----- oracle payoff matrix -------------------------------------------
  mu_selected <- matrix(0, K, N)
  for (i in 1:K) for (t in 1:N)
    mu_selected[i, t] <-
      if (dynamics == "independent")
        mu[i, z_true[i, t] + 1] else
          mu[i, z_true[t] + 1]
  oracle_arm <- apply(mu_selected, 2, which.max)
  
  ## ----- storage ---------------------------------------------------------
  observed_rewards <- vector("list", K)
  smoothed_models  <- vector("list", K)
  selected_arms    <- numeric(N)
  received_rewards <- numeric(N)
  regret           <- numeric(N)
  
  ## =============================== loop ==================================
  for (t in 1:N) {
    # -- re-fit KFAS every batch_size --------------------------------------
    if (t %% batch_size == 0) {
      for (k in 1:K) {
        yk <- observed_rewards[[k]]
        if (length(yk) >= 2) {
          y_pad <- rep(NA_real_, t)
          y_pad[seq_along(yk)] <- yk
          smoothed_models[[k]] <- fit_kfas_bernoulli_ar(y_pad)  # from earlier code
        }
      }
    }
    
    # -- Thompson draw -----------------------------------------------------
    sampled_values <- numeric(K)
    for (k in 1:K) {
      mod <- smoothed_models[[k]]
      if (!is.null(mod)) {
        n_k     <- length(observed_rewards[[k]])
        sigma   <- 0.1 / sqrt(max(1, n_k))      # shrinking exploration noise
        mu_last <- tail(mod$smoothed_logit_mu, 1)
        sampled_values[k] <- inv_logit(rnorm(1, mu_last, sigma))
      } else {
        sampled_values[k] <- runif(1)
      }
    }
    
    # -- observe reward & update ------------------------------------------
    a_t <- which.max(sampled_values)
    selected_arms[t]    <- a_t
    r_t                 <- y_true[a_t, t]
    received_rewards[t] <- r_t
    observed_rewards[[a_t]] <- c(observed_rewards[[a_t]], r_t)
    regret[t] <- mu_selected[oracle_arm[t], t] - mu_selected[a_t, t]
  }
  
  list(
    cumulative_reward = cumsum(received_rewards),
    cumulative_regret = cumsum(regret),
    selected_arms     = selected_arms,
    smoothed_models   = smoothed_models
  )
}