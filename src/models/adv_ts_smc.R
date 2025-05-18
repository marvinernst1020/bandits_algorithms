# ADAVANCED TS - SMC

thompson_advanced_smc <- function(K, N, y_true, z_true,
                                  batch_size = 100,
                                  n_particles = 200,
                                  dynamics = c("common","independent")) {
  dynamics <- match.arg(dynamics)
  S <- 2    # number of latent states
  
  # 1. Initialize particle cloud
  #   each particle m has:
  #     pi[m, 1:2]    -- row of transition prob from state 0 and 1
  #     mu[m, i, s]   -- emission prob for arm i in state s
  #     z_last[m]     -- last latent state (0 or 1)
  pi_particles <- matrix(0.5, nrow = n_particles, ncol = 2)
  mu_particles <- array(0.5, dim = c(n_particles, K, S))
  z_last       <- sample(0:(S-1), size = n_particles, replace = TRUE)
  weights      <- rep(1/n_particles, n_particles)
  
  # Storage
  selected_arms   <- integer(N)
  received_rewards <- numeric(N)
  cumulative_regret <- numeric(N)
  
  # Oracle for regret
  mu_selected <- matrix(0, nrow = K, ncol = N)
  for (t in 1:N) {
    for (i in 1:K) {
      if (dynamics=="common") {
        mu_selected[i,t] <- mu_particles[1,i, z_true[t]+1]
      } else {
        mu_selected[i,t] <- mu_particles[1,i, z_true[i,t]+1]
      }
    }
  }
  oracle_arm <- apply(mu_selected, 2, which.max)
  
  # 2. TS loop with SMC updates
  for (t in seq_len(N)) {
    # 2a. At batch boundaries, update particle weights with the last batch  
    if (t >= batch_size && (t %% batch_size)==1) {
      start_idx <- t - batch_size
      batch_idx <- start_idx:(t-1)
      y_batch   <- y_true[, batch_idx, drop=FALSE]
      
      # 2a.i. For each particle, compute weight = P(y_batch | particle)
      logw <- numeric(n_particles)
      for (m in seq_len(n_particles)) {
        logp <- 0
        z <- z_last[m]
        for (u in seq_along(batch_idx)) {
          # transition
          if (u>1) {
            prob_z <- if (z==0) pi_particles[m,1] else pi_particles[m,2]
            z <- rbinom(1,1,prob_z)
          }
          # emission likelihood across all arms
          for (i in 1:K) {
            p <- mu_particles[m,i,z+1]
            y <- y_batch[i,u]
            logp <- logp + dbinom(y, size=1, prob=p, log=TRUE)
          }
        }
        logw[m] <- log(weights[m]) + logp
      }
      # normalize weights
      maxlw     <- max(logw)
      weights   <- exp(logw - maxlw)
      weights  <- weights / sum(weights)
      
      # 2a.ii. Resample particles
      idx       <- sample(seq_len(n_particles), n_particles, replace=TRUE, prob=weights)
      pi_particles <- pi_particles[idx, , drop=FALSE]
      mu_particles <- mu_particles[idx,, , drop=FALSE]
      z_last       <- z_last[idx]
      weights      <- rep(1/n_particles, n_particles)
      
      # 2a.iii. Rejuvenate static parameters via one-step Gibbs on each particle
      #       (use conjugacy: Beta priors + Bernoulli counts from this batch)
      for (m in seq_len(n_particles)) {
        # Count transitions in this particle's inferred z-path
        # We simulate a path again under its pi to get transition counts
        z <- z_last[m]
        n00 <- n01 <- n10 <- n11 <- 0
        # Emission counts
        # We approximate by using the expected counts at the MAP z-path:
        # Here we just use the last latent state to update mu for that state:
        s_counts <- matrix(0, nrow=K, ncol=S)  # successes
        f_counts <- matrix(0, nrow=K, ncol=S)  # failures
        
        for (u in seq_along(batch_idx)) {
          if (u>1) {
            prob_z <- if (z==0) pi_particles[m,1] else pi_particles[m,2]
            z_prev <- z
            z <- rbinom(1,1,prob_z)
            if      (z_prev==0 && z==0) n00 <- n00+1
            else if (z_prev==0 && z==1) n01 <- n01+1
            else if (z_prev==1 && z==0) n10 <- n10+1
            else                         n11 <- n11+1
          }
          for (i in 1:K) {
            y <- y_batch[i,u]
            if (y==1) s_counts[i,z+1] <- s_counts[i,z+1] + 1
            else      f_counts[i,z+1] <- f_counts[i,z+1] + 1
          }
        }
        # Update pi row-wise with Beta(1+success,1+failure)
        pi_particles[m,1] <- rbeta(1, 1 + n01, 1 + n00)  # P(z=1|prev=0)
        pi_particles[m,2] <- rbeta(1, 1 + n11, 1 + n10)  # P(z=1|prev=1)
        
        # Update mu for each arm-state
        for (i in 1:K) {
          for (s in 1:S) {
            mu_particles[m,i,s] <- rbeta(
              1,
              1 + s_counts[i,s],
              1 + f_counts[i,s]
            )
          }
        }
      }
    }
    
    # 2b. Thompson sample one particle
    m_draw <- sample(seq_len(n_particles), 1, prob=weights)
    pi_draw <- pi_particles[m_draw, ]
    mu_draw <- mu_particles[m_draw,, ]
    z       <- z_last[m_draw]
    
    # 2c. Sample next latent state and pick arm
    next_p   <- if (z==0) pi_draw[1] else pi_draw[2]
    z_t      <- rbinom(1,1,next_p)
    selected <- which.max(mu_draw[,z_t+1])
    
    # 2d. Observe reward
    r_t <- y_true[selected, t]
    received_rewards[t] <- r_t
    selected_arms[t] <- selected
    z_last[m_draw] <- z_t
    
    # 2e. Regret
    true_z   <- if (dynamics=="common") z_true[t] else z_true[selected,t]
    cumulative_regret[t] <- (mu_selected[oracle_arm[t],t] -
                               mu[selected, true_z+1])
  }
  
  list(
    selected_arms    = selected_arms,
    cumulative_reward = cumsum(received_rewards),
    cumulative_regret = cumulative_regret
  )
}
