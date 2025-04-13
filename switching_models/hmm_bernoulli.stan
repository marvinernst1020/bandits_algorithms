//
// HMM - BERNOULLI
//

data {
  int<lower=1> N;
  int<lower=0,upper=1> y[N];
}
parameters {
  simplex[2] init;
  simplex[2] trans[2];
  real alpha;
  real<lower=0> delta;
}
transformed parameters {
  real theta[2];
  theta[1] = inv_logit(alpha);
  theta[2] = inv_logit(alpha + delta);
}
model {
  vector[2] log_alpha;
  matrix[2, N] log_prob;

  alpha ~ normal(0, 1);
  delta ~ normal(2, 1); 
  trans[1] ~ dirichlet([10, 1]);
  trans[2] ~ dirichlet([1, 10]);

  for (k in 1:2)
    log_prob[k,1] = log(init[k]) + bernoulli_lpmf(y[1] | theta[k]);

  for (t in 2:N) {
    for (k in 1:2) {
      vector[2] logsum;
      for (j in 1:2) {
        logsum[j] = log_prob[j,t-1] + log(trans[j,k]);
      }
      log_prob[k,t] = log_sum_exp(logsum) + bernoulli_lpmf(y[t] | theta[k]);
    }
  }

  target += log_sum_exp(log_prob[,N]);
}

