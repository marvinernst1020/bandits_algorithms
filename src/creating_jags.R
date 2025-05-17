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



