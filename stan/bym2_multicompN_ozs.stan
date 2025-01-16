functions {
  /**
   * Unconstrain component-wise sum-to-zero vector
   *
   * @param phi constrained zero-sum parameters
   * @param idxs component start and end indices
   * @param sizes component sizes
   * @return vector phi_ozs, the vector whose slices sum to zero
   */
  vector zero_sum_components_lp(vector phi, array[ , ] int idxs, array[] int sizes) {
    vector[sum(sizes)] phi_ozs;
    int idx_phi = 1;
    int idx_ozs = 1;
    for (i in 1:size(sizes)) {
      phi_ozs[idx_ozs : idx_ozs + sizes[i] - 1]
	= zero_sum_constrain_lp(segment(phi, idx_phi, sizes[i] - 1));
      idx_phi += sizes[i] - 1;
      idx_ozs += sizes[i];
    }
    return phi_ozs;
  }

  /**
   * Unconstrain component-wise sum-to-zero vector and adjust for understimation of variance.
   *
   * @param y constrained zero-sum parameters
   * @return vector z, the vector whose slices sum to zero
   */
  vector zero_sum_constrain_lp(vector y) {
    int N = num_elements(y);
    vector[N + 1] z = zeros_vector(N + 1);
    real sum_w = 0;
    for (ii in 1:N) {
      int i = N - ii + 1; 
      real n = i;
      real w = y[i] * inv_sqrt(n * (n + 1));
      sum_w += w;
      z[i] += sum_w;     
      z[i + 1] -= w * n;    
    }
    target += normal_lpdf(z | 0, inv_sqrt(1 - inv(N + 1)));
    return z;
  }
}

data {
  int<lower=0> N;
  array[N] int<lower=0> y; // count outcomes
  vector<lower=0>[N] E; // exposure
  int<lower=1> K; // num covariates
  matrix[N, K] xs; // design matrix

  int<lower=0, upper=N> N_components;
  array[N_components] int<lower=1, upper=N> component_sizes;
  vector<lower=0>[N_components] scaling_factors;
  int<lower=0, upper=N> N_singletons;

  // neighbor graph structure
  int<lower = 0> N_edges;  // number of neighbor pairs
  array[2, N_edges] int<lower = 1, upper = (N - N_singletons)> neighbors;  // columnwise adjacent
}
transformed data {
  vector[N] log_E = log(E);
  // center continuous predictors 
  vector[K] means_xs;  // column means of xs before centering
  matrix[N, K] xs_centered;  // centered version of xs
  for (k in 1:K) {
    means_xs[k] = mean(xs[, k]);
    xs_centered[, k] = xs[, k] - means_xs[k];
  }
  // check spatial data structure
  int N_connected = sum(component_sizes);
  if (N != N_connected + N_singletons) {
    reject("N must equal sum(component_sizes) + N_singletons",
	   "N = ", N, " sum(component_sizes) = ", N_connected,
	   "N_singletons = ", N_singletons);
  }
  // compute indices, vector of scaling factors
  vector<lower=0>[N_connected] taus;
  array[N_components, 2] int component_idxs;
  int idx = 1;
  for (n in 1:N_components) {
    component_idxs[n, 1] = idx;
    component_idxs[n, 2] = idx + component_sizes[n] - 1;
    for (j in 1:component_sizes[n]) {
      taus[idx] = scaling_factors[n];
      idx += 1;
    }
  }
}
parameters {
  real beta0;  // intercept
  vector[K] betas;  // covariates
  
  real<lower=0> sigma;  // random effects scale
  real<lower=0, upper=1> rho;  // proportion unstructured vs. spatially structured variance
  
  vector[N_connected] theta; // heterogeneous effects
  vector[N_singletons] singletons_re; // random effects for areas with no neighbours

  // concatenation of unconstrained vectors, each size N - 1
  vector[N_connected - N_components] phi_raw;
}
transformed parameters {
  vector[N_connected] phi = zero_sum_components_lp(phi_raw, component_idxs, component_sizes);
  vector[N] gamma;  // convolved spatial random effect - Riebler BYM2
  gamma[1 : N_connected] =
    sqrt(1 - rho) * theta + sqrt(rho * inv(taus)) .* phi;
  gamma[N_connected + 1 : N] = singletons_re;
}
model {
  y ~ poisson_log(log_E + beta0 + xs_centered * betas + gamma * sigma);
  beta0 ~ std_normal();
  betas ~ std_normal();
  theta ~ std_normal();
  singletons_re ~ std_normal();
  sigma ~ std_normal();
  rho ~ beta(0.5, 0.5);
  target += -0.5 * dot_self(phi[neighbors[1]] - phi[neighbors[2]]); // ICAR
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
