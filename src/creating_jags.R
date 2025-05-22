###### CREATING JAGS MODELS #######

# set the correct working directory:
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))

##### Poor Model:

cat(
  'model {
  y[1] ~ dbern(p[1])
  z[1] ~ dbern(0.5)
  p[1] <- z[1]*mu1 + (1 - z[1])*mu0

  for (t in 2:N) {
    z[t] ~ dbern(z[t-1]*pi[2] + (1 - z[t-1])*pi[1])
    p[t] <- z[t]*mu1 + (1 - z[t])*mu0
    y[t] ~ dbern(p[t])
  }

  mu0 ~ dbeta(1,1)
  mu1 ~ dbeta(1,1)
  pi[1] ~ dbeta(1,1)
  pi[2] ~ dbeta(1,1)
}
', file = "models/poor_model.jags")


##### Advanced Model:

cat(
  'model {
  for (i in 1:K) {
    for (t in 1:N) {
      p_arm[i,t] <- z[t] * mu[i,2] + (1 - z[t]) * mu[i,1]
    }
  }

  z[1] ~ dbern(0.5)
  for (i in 1:K) {
    y_obs[i,1] ~ dbern(p_arm[i,1])
  }

  for (t in 2:N) {
    z[t] ~ dbern(z[t-1] * pi[2] + (1 - z[t-1]) * pi[1])
    for (i in 1:K) {
      y_obs[i,t] ~ dbern(p_arm[i,t])
    }
  }

  pi[1] ~ dbeta(1,1)
  pi[2] ~ dbeta(1,1)

  for (i in 1:K) {
    for (s in 1:2) {
      mu[i,s] ~ dbeta(1,1)
    }
  }
}
', file = "models/advanced_model.jags")


##### AR Model:

cat(
  "
  model {
    logit_mu[1] <- logit(mu1)
    p[1] <- mu1
    y[1] ~ dbern(p[1])

    for (t in 2:N) {
      eps[t]       ~ dnorm(0, tau)                    
      logit_mu[t]  <- phi * logit_mu[t-1] + eps[t]
      p[t]  <- ilogit(logit_mu[t])
      y[t]  ~ dbern(p[t])
    }

    # priors
    mu1 ~ dbeta(1, 1)
    phi ~ dnorm(0, 1 / pow(0.5,2)) T(-1,1)
    tau ~ dgamma(0.001, 0.001)     
  }
  ",
  file = "models/ar_model.jags"
)
## or logit_mu[t] ~ dnorm(phi * logit_mu[t-1], tau)
