# ADVANCED TS - NIMBLE VERSION 

# 1. Define Nimble HMM model (same structure as in JAGS)
hmm_code <- nimbleCode({
  for (i in 1:K) {
    for (t in 1:N) {
      p_arm[i, t] <- z[t] * mu[i, 2] + (1 - z[t]) * mu[i, 1]
      y_obs[i, t] ~ dbern(p_arm[i, t])
    }
  }
  z[1] ~ dbern(0.5)
  for (t in 2:N) {
    z[t] ~ dbern(z[t - 1] * pi[2] + (1 - z[t - 1]) * pi[1])
  }
  pi[1] ~ dbeta(1, 1)
  pi[2] ~ dbeta(1, 1)
  for (i in 1:K) {
    mu[i, 1] ~ dbeta(1, 1)
    mu[i, 2] ~ dbeta(1, 1)
  }
})

# 2. Compile Nimble model and MCMC setup
compile_nimble_model <- function(K, window_size) {
  constants <- list(K = K, N = window_size)
  data <- list(y_obs = matrix(NA, nrow = K, ncol = window_size))
  inits <- list(
    pi = rep(0.5, 2),
    mu = matrix(0.5, nrow = K, ncol = 2),
    z  = rep(0, window_size)
  )
  model  <- nimbleModel(hmm_code, constants, data, inits, calculate = FALSE)
  cmodel <- compileNimble(model)
  conf   <- configureMCMC(model, monitors = c("pi", "mu", "z"))
  mcmc   <- buildMCMC(conf)
  cmcmc  <- compileNimble(mcmc, project = model)
  list(model = model, cmodel = cmodel, cmcmc = cmcmc)
}

# 3. Thompson Sampling procedure using Nimble
thompson_advanced_nimble <- function(K, N, y_true, z_true, mu,
                                     batch_size   = 100,
                                     window_size  = 1000,
                                     burn         = 500,
                                     n_iter       = 200,
                                     dynamics     = c("common", "independent")) {
  
  dynamics <- match.arg(dynamics)
  first_run <- TRUE
  
  # Oracle
  mu_selected <- matrix(0, nrow = K, ncol = N)
  for (t in 1:N) {
    for (i in 1:K) {
      mu_selected[i, t] <- if (dynamics == "common") mu[i, z_true[t] + 1] else mu[i, z_true[i, t] + 1]
    }
  }
  oracle_arm <- apply(mu_selected, 2, which.max)
  
  # Compile Nimble model
  nimble_objs <- compile_nimble_model(K, window_size)
  cmodel <- nimble_objs$cmodel
  cmcmc  <- nimble_objs$cmcmc
  
  # Tracking
  selected_arms    <- integer(N)
  received_rewards <- numeric(N)
  regret           <- numeric(N)
  post_pi          <- rep(0.5, 2)
  post_mu          <- matrix(0.5, nrow = K, ncol = 2)
  last_z           <- sample(0:1, 1)
  
  # t = 1
  selected_arms[1]    <- sample(1:K, 1)
  received_rewards[1] <- y_true[selected_arms[1], 1]
  true_z1             <- if (dynamics == "common") z_true[1] else z_true[selected_arms[1], 1]
  regret[1]           <- mu_selected[oracle_arm[1], 1] - mu[selected_arms[1], true_z1 + 1]
  
  for (t in 2:N) {
    
    if (t %% batch_size == 1 && t > window_size) {
      window    <- min(window_size, t - 1)
      start_idx <- t - window
      y_block   <- y_true[, start_idx:(t - 1), drop = FALSE]
      
      cmodel$y_obs <- y_block
      cmodel$z     <- c(tail(cmodel$z, window - 1), last_z)
      
      if (first_run) {
        cmcmc$run(burn + n_iter, reset = TRUE)
        first_run <- FALSE
      } else {
        cmcmc$run(n_iter, reset = TRUE)
      }
      
      samples <- as.matrix(cmcmc$mvSamples)
      draw    <- sample(nrow(samples), 1)
      post_pi <- samples[draw, c("pi[1]", "pi[2]")]
      mu_cols <- grep("^mu", colnames(samples))
      post_mu <- matrix(samples[draw, mu_cols], nrow = K, byrow = TRUE)
      last_z  <- samples[draw, paste0("z[", window, "]")]
      
      message(glue::glue("[{Sys.time()}] Performed inference at time step {t} (window {start_idx}:{t-1})"))
    }
    
    # Draw latent z and select arm
    p_trans <- post_pi[if (last_z == 0) 1 else 2]
    z_t     <- rbinom(1, 1, p_trans)
    scores  <- post_mu[, z_t + 1]
    selected <- which.max(scores)
    
    selected_arms[t]    <- selected
    received_rewards[t] <- y_true[selected, t]
    last_z              <- z_t
    
    true_z <- if (dynamics == "common") z_true[t] else z_true[selected, t]
    regret[t] <- mu_selected[oracle_arm[t], t] - mu[selected, true_z + 1]
  }
  
  list(
    selected_arms     = selected_arms,
    cumulative_reward = cumsum(received_rewards),
    cumulative_regret = cumsum(regret)
  )
}
