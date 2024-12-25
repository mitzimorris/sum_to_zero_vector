// multi-level model for binomial data with 4 categorical predictors.
data {
  int<lower=1> N; // number of strata
  int<lower=1> N_age;
  int<lower=1> N_eth;
  int<lower=1> N_edu;

  array[N] int<lower=0> pos_tests;
  array[N] int<lower=0> tests;
  array[N] int<lower=1, upper=2> sex;
  array[N] int<lower=1, upper=N_age> age;
  array[N] int<lower=1, upper=N_eth> eth;
  array[N] int<lower=1, upper=N_edu> edu;

  // hyperparameters
  real<lower=0, upper=1> sens;
  real<lower=0, upper=1> spec;
  real intercept_prior_mean;
  real intercept_prior_scale;
}
transformed data {
  real mean_sex = mean(sex);
  vector[N] sex_c = to_vector(sex) - mean_sex;
}
parameters {
  real beta_0;
  real beta_sex;
  real<lower=0> sigma_age, sigma_eth, sigma_edu;
  sum_to_zero_vector[N_age] beta_age;
  sum_to_zero_vector[N_eth] beta_eth;
  sum_to_zero_vector[N_edu] beta_edu;
}
transformed parameters {
  // non-standard link function
  vector[N] p =  inv_logit(beta_0 + beta_sex * sex_c + beta_age[age]
			   + beta_eth[eth] +  beta_edu[edu]);
  vector[N] p_sample = p * sens + (1 - p) * (1 - spec);
}
model {
  pos_tests ~ binomial(tests, p_sample);  // likelihood

  // priors
  beta_0 ~ normal(intercept_prior_mean, intercept_prior_scale);
  beta_sex ~ std_normal();
  beta_age ~ normal(0, sigma_age);
  beta_eth ~ normal(0, sigma_eth);
  beta_edu ~ normal(0, sigma_edu);
  sigma_eth ~ std_normal();   // need to account for sum-to-zero
  sigma_age ~ std_normal();
  sigma_edu ~ std_normal();
}
generated quantities {
  real beta_intercept = beta_0 - mean_sex * beta_sex;
  array[N] int<lower=0>y_rep = binomial_rng(tests, p_sample);
}
