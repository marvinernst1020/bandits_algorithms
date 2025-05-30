###### ADDITIONAL MODELS ######

# set the correct working directory:
setwd(dirname(rstudioapi::getActiveDocumentContext()$path))


##### Advanced Model:

cat(
  'model {
    for (i in 1:K) {
      for (t in 1:N) {
        p_arm[i,t] <- z[t] * mu[i,2] + (1 - z[t]) * mu[i,1]
        y_obs[i,t] ~ dbern(p_arm[i,t])
      }
    }
    z[1] ~ dbern(0.5)
    for (t in 2:N) {
      z[t] ~ dbern(z[t-1] * pi[2] + (1 - z[t-1]) * pi[1])
    }

    # instead of pi[i] ~ dbeta(1,1)
    pi[1] ~ dbeta(20,  1)  
    pi[2] ~ dbeta(20,  1)  

    # IDENTIFIABILITY CONSTRAINTS:
    mu[1,1] ~ dbeta(1,1)
    mu[2,1] ~ dbeta(1,1) T(mu[1,1], 1)  

    mu[1,2] ~ dbeta(1,1)
    mu[2,2] ~ dbeta(1,1) T(mu[1,2], 1)  
  }
', file = "models/advanced_model.jags")



