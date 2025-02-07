data {
  int<lower=0> N;
  array[N] int<lower=0> y; // count outcomes
  vector<lower=0>[N] E; // exposure
  int<lower=1> K; // num covariates
  matrix[N, K] xs; // design matrix

  // spatial structure
  int<lower = 0> N_edges;  // number of neighbor pairs
  array[2, N_edges] int<lower = 1, upper = N> neighbors;  // columnwise adjacent
}
transformed data {
  vector[N] log_E = log(E);
  // center continuous predictors 
  vector[K] means_xs;
  matrix[N, K] xs_centered;
  for (k in 1:K) {
    means_xs[k] = mean(xs[, k]);
    xs_centered[, k] = xs[, k] - means_xs[k];
  }
}
parameters {
  real beta0; // intercept
  vector[K] betas; // covariates
  sum_to_zero_vector[N] phi; // spatial effects
  real<lower=0> sigma; // overall spatial variance
}
model {
  y ~ poisson_log(log_E + beta0 + xs_centered * betas + phi * sigma);
  beta0 ~ std_normal();
  betas ~ std_normal();
  sigma ~ std_normal();
  target += -0.5 * dot_self(phi[neighbors[1]] - phi[neighbors[2]]);  // ICAR prior
}
generated quantities {
  real beta_intercept = beta0 - dot_product(means_xs, betas);  // adjust intercept
  array[N] int y_rep;
  {
    vector[N] eta = log_E + beta0 + xs_centered * betas + phi * sigma;
    y_rep = max(eta) < 26 ? poisson_log_rng(eta) : rep_array(-1, N);
  }
}
