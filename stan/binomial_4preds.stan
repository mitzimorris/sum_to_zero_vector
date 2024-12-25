data {
  int<lower=1> N_age;
  int<lower=1> N_eth;
  int<lower=1> N_edu;
  int<lower=1, upper=(2 * N_age * N_eth * N_edu)> N;

  // binomial observations
  array[N] int<lower=0> tests; // per-stratum trials
  array[N] int<lower=0> cases; // per-stratum number of successes

  // categorical predictors
  array[N] int<lower=1, upper=2> sex;
  array[N] int<lower=1, upper=N_age> age;
  array[N] int<lower=1, upper=N_eth> eth;
  array[N] int<lower=1, upper=N_edu> edu;

  // hyperparameters
  real<lower=0, upper=1> sens;
  real<lower=0, upper=1> spec;
  real intercept_prior_mean;
  real<lower=0> intercept_prior_scale;
}
parameters {
  real alpha;
  real alpha_female;
  real<lower=0> sigma_age;
  real<lower=0> sigma_eth;
  real<lower=0> sigma_edu;
  vector<multiplier=sigma_age>[N_age] a_age;  // non-centered parameterization
  vector<multiplier=sigma_eth>[N_eth] a_eth;
  vector<multiplier=sigma_edu>[N_edu] a_edu;
}
transformed parameters {
  // non-standard link function
  vector[N] p =  inv_logit(alpha + alpha_female * sex + a_age[age] + a_eth[eth] +  a_edu[edu]);
  vector[N] p_sample = p * sens + (1 - p) * (1 - spec);
}
model {
  cases ~ binomial(tests, p_sample);  // likelihood

  // priors
  alpha ~ normal(intercept_prior_mean, intercept_prior_scale);
  alpha_female ~ std_normal();
  a_age ~ normal(0, sigma_age);
  a_eth ~ normal(0, sigma_eth);
  a_edu ~ normal(0, sigma_edu);
  sigma_eth ~ std_normal();
  sigma_age ~ std_normal();
  sigma_edu ~ std_normal();
}
generated quantities {
  array[N] int<lower=0>y_rep = binomial_rng(tests, p_sample);
}
