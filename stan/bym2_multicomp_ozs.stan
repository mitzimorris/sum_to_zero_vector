data {
  int<lower=0> N;
  array[N] int<lower=0> y; // count outcomes
  vector<lower=0>[N] E; // exposure
  int<lower=1> K; // num covariates
  matrix[N, K] xs; // design matrix

  int<lower=0, upper=N> N_singletons;
  // N_components fixed, not data.
  array[6] int<lower=1, upper=N> component_sizes;
  vector<lower=0>[6] scaling_factors;

  // neighbor graph structure
  int<lower = 0> N_edges;  // number of neighbor pairs
  array[2, N_edges] int<lower = 1, upper = (N - N_singletons)> neighbors;  // columnwise adjacent
}
transformed data {
  int N_components = 6;
  int N_connected = N - N_singletons;

  // compute indices, vector of scaling factors
  vector<lower=0>[N_connected] taus;
  array[N_components] int beg_idx;
  array[N_components] int end_idx;
  int idx = 1;
  for (n in 1:N_components) {
    beg_idx[n] = idx;
    end_idx[n] = beg_idx[n] + component_sizes[n] - 1;
    for (j in 1:component_sizes[n]) {
      taus[idx] = scaling_factors[n];
      idx += 1;
    }
  }

  vector[N] log_E = log(E);
  // center continuous predictors 
  vector[K] means_xs;  // column means of xs before centering
  matrix[N, K] xs_centered;  // centered version of xs
  for (k in 1:K) {
    means_xs[k] = mean(xs[, k]);
    xs_centered[, k] = xs[, k] - means_xs[k];
  }
}
parameters {
  real beta0;  // intercept
  vector[K] betas;  // covariates
  
  real<lower=0> sigma;  // random effects scale
  real<lower=0, upper=1> rho;  // proportion unstructured vs. spatially structured variance
  
  vector[N_connected] theta; // heterogeneous effects
  vector[N_singletons] singletons_re; // random effects for areas with no neighbours
  // necessary because we can't create ragged arrays
  sum_to_zero_vector[component_sizes[1]] phi_1;
  sum_to_zero_vector[component_sizes[2]] phi_2;
  sum_to_zero_vector[component_sizes[3]] phi_3;
  sum_to_zero_vector[component_sizes[4]] phi_4;
  sum_to_zero_vector[component_sizes[5]] phi_5;
  sum_to_zero_vector[component_sizes[6]] phi_6;
}
transformed parameters {
  vector[N_connected] phi;
  phi[beg_idx[1]:end_idx[1]] = phi_1;
  phi[beg_idx[2]:end_idx[2]] = phi_2;
  phi[beg_idx[3]:end_idx[3]] = phi_3;
  phi[beg_idx[4]:end_idx[4]] = phi_4;
  phi[beg_idx[5]:end_idx[5]] = phi_5;
  phi[beg_idx[6]:end_idx[6]] = phi_6;
  vector[N] gamma;  // convolved spatial random effect - Riebler BYM2
  gamma[1 : N_connected] =
    sqrt(1 - rho) * theta + sqrt(rho * inv(taus)) .* phi;
  gamma[N_connected + 1 : N] = singletons_re;
}
model {
  y ~ poisson_log(log_E + beta0 + xs_centered * betas + gamma * sigma);
  rho ~ beta(0.5, 0.5);
  target += -0.5 * dot_self(phi[neighbors[1]] - phi[neighbors[2]]); // ICAR
  beta0 ~ std_normal();
  betas ~ std_normal();
  theta ~ std_normal();
  sigma ~ std_normal();
  singletons_re ~ std_normal();
}
generated quantities {
  real beta0_intercept = beta0 - dot_product(means_xs, betas);  // adjust intercept
  array[N] int y_rep;
 {
   vector[N] eta = log_E + beta0 + xs_centered * betas + gamma * sigma;
   if (max(eta) > 21) {
     // avoid overflow in poisson_log_rng
     print("max eta too big: ", max(eta));
     for (n in 1:N) {
	y_rep[n] = -1;
     }
   } else {
     for (n in 1:N) {
	y_rep[n] = poisson_log_rng(eta[n]);
     }
   }
 }
}