#### AR with Thompson Sampling - Using KFAS

# helper functions:
logit <- function(p) log(p / (1 - p))
inv_logit <- function(x) exp(x) / (1 + exp(x))

# Posterior Inference function:
fit_kfas_bernoulli_ar <- function(y) {
  model <- SSModel(y ~ SSMtrend(degree = 1, Q = list(1e-4)), distribution = "binomial") # this bypasses optimization and uses a small fixed process noise - but if we do this we have to mention this in the report!!
  fit <- fitSSM(model, inits = log(0.1), method = "BFGS")
  out <- KFS(fit$model, smoothing = "state")
  list(model = fit$model, smoothed_logit_mu = out$alphahat[, 1])
}

# TS using KFAS AR(1) Inference:
thompson_ar_kfas <- function(K, N, mu, y_true, z_true, batch_size) {
  observed_rewards <- vector("list", K)
  selected_arms <- numeric(N)
  received_rewards <- numeric(N)
  regret <- numeric(N)
  mu_selected <- matrix(0, nrow = K, ncol = N)
  
  # Generate oracle rewards
  for (i in 1:K) {
    for (t in 1:N) {
      mu_selected[i, t] <- if (is.matrix(z_true)) mu[i, z_true[i, t] + 1] else mu[i, z_true[t] + 1]
    }
  }
  oracle_arm <- apply(mu_selected, 2, which.max)
  
  smoothed_models <- rep(list(NULL), K)
  
  for (t in 1:N) {
    if (t %% batch_size == 0) {
      for (k in 1:K) {
        y_k <- observed_rewards[[k]]
        if (length(y_k) >= 2) {  # KFAS needs >= 2 obs, because it does not work when you only have 1 observation
          smoothed_models[[k]] <- fit_kfas_bernoulli_ar(y_k)
        }
      }
    }
    
    sampled_values <- numeric(K)
    for (k in 1:K) {
      mod <- smoothed_models[[k]]
      if (!is.null(mod)) {
        # take last smoothed logit_mu and add noise to simulate TS:
        logit_mu_last <- tail(mod$smoothed_logit_mu, 1)
        logit_mu_draw <- rnorm(1, mean = logit_mu_last, sd = 0.1)  # small noise, the posterior we add
        sampled_values[k] <- inv_logit(logit_mu_draw)
      } else {
        sampled_values[k] <- runif(1)
      }
    }
    
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
    selected_arms = selected_arms,
    smoothed_models = smoothed_models
  )
}

