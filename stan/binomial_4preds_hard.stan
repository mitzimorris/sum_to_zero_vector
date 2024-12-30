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
}
parameters {
  real beta_0;
  real beta_sex_raw;
  real<lower=0> sigma_age, sigma_eth, sigma_edu;
  vector[N_age - 1] beta_age_raw;
  vector[N_eth - 1] beta_eth_raw;
  vector[N_edu - 1] beta_edu_raw;
}
transformed parameters {
  vector[2] beta_sex = [beta_sex_raw, -beta_sex_raw]';

  // non-centered parameterization
  vector<multiplier=sigma_age>[N_age] beta_age = append_row(beta_age_raw, -sum(beta_age_raw));
  vector<multiplier=sigma_eth>[N_eth] beta_eth = append_row(beta_eth_raw, -sum(beta_eth_raw));
  vector<multiplier=sigma_edu>[N_edu] beta_edu = append_row(beta_edu_raw, -sum(beta_edu_raw));

  vector[N] eta =  inv_logit(beta_0 + beta_sex[sex] + beta_age[age] + beta_eth[eth] +  beta_edu[edu]);
  vector[N] prob_pos_test = eta * sens + (1 - eta) * (1 - spec);
}
model {
  pos_tests ~ binomial(tests, prob_pos_test);  // likelihood

  // priors
  beta_0 ~ normal(0, 10);
  beta_sex ~ normal(0, 2.5);
  beta_age ~ normal(0, sigma_age);
  beta_eth ~ normal(0, sigma_eth);
  beta_edu ~ normal(0, sigma_edu);
  sigma_eth ~ normal(0, 2.5);
  sigma_age ~ normal(0, 2.5);
  sigma_edu ~ normal(0, 2.5);
}
generated quantities {
  array[N] int<lower=0>y_rep = binomial_rng(tests, prob_pos_test);
}
