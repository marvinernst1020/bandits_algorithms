//
// MSM - BERNOULLI
// 

data {
  int<lower=1> N;
  int<lower=0,upper=1> y[N];
}
parameters {
  simplex[2] pi;               // State probabilities
  real<lower=0, upper=1> p[2]; // Bernoulli success probabilities for each state
}
model {
  for (t in 1:N) {
    target += log_mix(pi[1],
                      bernoulli_lpmf(y[t] | p[1]),
                      bernoulli_lpmf(y[t] | p[2]));
  }
}

