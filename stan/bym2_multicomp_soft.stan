functions {
  /**
   * Compute ICAR, use soft-sum-to-zero constraint for identifiability
   *
   * @param phi vector of varying effects
   * @param adjacency parallel arrays of indexes of adjacent elements of phi
   * @param epsilon allowed variance for soft centering
   * @return ICAR log probability density
   * @reject if the the adjacency matrix does not have two rows
   */
  real standard_icar_lpdf(vector phi, array[ , ] int adjacency, array [] int sizes, real epsilon) {
    real result = -0.5 * dot_self(phi[adjacency[1]] - phi[adjacency[2]]);
    int N_components = size(sizes);
    int idx = 1;
    for (n in 1:N_components) {
      result += normal_lupdf(sum(segment(phi, idx, sizes[n])) | 0, epsilon * sizes[n]);
      idx += sizes[n];
    }
    return result;
  }
}
data {
  int<lower=0> N;
  array[N] int<lower=0> y; // count outcomes
  vector<lower=0>[N] E; // exposure
  int<lower=1> K; // num covariates
  matrix[N, K] xs; // design matrix

  // neighbor graph structure
  int<lower=0, upper=N> N_components;
  array[N_components] int<lower=1, upper=N> component_sizes;
  int<lower=0, upper=N> N_singletons;
  int<lower = 0> N_edges;  // number of neighbor pairs

  array[2, N_edges] int<lower = 1, upper = (N - N_singletons)> neighbors;  // columnwise adjacent
  vector<lower=0>[N_components] scaling_factors;
}
transformed data {
  int N_connected = N - N_singletons;

  // compute indices, vector of scaling factors
  vector<lower=0>[N_connected] taus;
  int idx = 1;
  for (n in 1:N_components) {
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
  vector[N_connected] phi;
  vector[N_singletons] singletons_re; // random effects for areas with no neighbours
}
transformed parameters {
  vector[N] gamma;  // convolved spatial random effect - Riebler BYM2
  gamma[1 : N_connected] =
    sqrt(1 - rho) * theta + sqrt(rho * inv(taus)) .* phi;
  gamma[N_connected + 1 : N] = singletons_re;
}
model {
  y ~ poisson_log(log_E + beta0 + xs_centered * betas + gamma * sigma);
  rho ~ beta(0.5, 0.5);
  phi ~ standard_icar(neighbors, component_sizes, 0.001);
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
