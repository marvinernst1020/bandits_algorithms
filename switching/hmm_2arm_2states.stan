// File: hmm_2arm_2states.stan

data {
  int<lower=1> N;
  int<lower=0, upper=1> y[N]; 
  int<lower=1, upper=2> arm[N]; 
}

parameters {
  simplex[2] init; 
  simplex[2] trans[2]; 
  real<lower=0, upper=1> theta[2, 2]; 
}

model {
  matrix[2, N] log_prob;

  // Priors
  for (i in 1:2)
    for (j in 1:2)
      theta[i, j] ~ beta(2, 2); // prior on success probabilities

  // Forward algorithm
  for (k in 1:2)
    log_prob[k,1] = log(init[k]) + bernoulli_lpmf(y[1] | theta[k, arm[1]]);

  for (t in 2:N) {
    for (k in 1:2) {
      vector[2] logsum;
      for (j in 1:2) {
        logsum[j] = log_prob[j, t-1] + log(trans[j, k]);
      }
      log_prob[k,t] = log_sum_exp(logsum) + bernoulli_lpmf(y[t] | theta[k, arm[t]]);
    }
  }

  target += log_sum_exp(log_prob[, N]);
}

