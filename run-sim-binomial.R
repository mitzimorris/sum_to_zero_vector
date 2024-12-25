library(dplyr)
library(cmdstanr)

source("gen_binom_zip_data.R")

sens = 0.7
spec = 0.99
log_baseline_prev = -7

N_tests = 100000
N_time = 50

pct_sex = c(0.42, 0.58)
beta_sex = c(0.15, -0.15)

N_age = 5
pct_age = c(0.3, 0.12, 0.15, 0.25, 0.35)
beta_age = c(-0.3, -0.2, 0.1, 0.25, 0.4)

N_eth = 3
pct_eth = c(0.7, 0.22, 0.08)
beta_eth = c(-0.3, 0.2, 0.1)

N_zip = 10
pct_zip = nprobs(N_zip);
sd_adi = 0.1
sd_edu = 0.06
sd_emp = 0.05
sd_inc = 0.05
sd_pov = 0.07
sd_urb = 0.03
beta_zip = generate_zip_beta(N_zip, sd_adi, sd_edu, sd_emp,
                             sd_inc, sd_pov, sd_urb)


sim_dataset = generate_binom_data(N_tests, N_age, N_eth, N_time, N_zip,
                                  pct_sex, pct_age, pct_eth, pct_zip,
                                  beta_sex, beta_age, beta_eth, beta_zip,
                                  sens, spec, log_baseline_prev)


df_cts = data.frame("x"=seq(sim_dataset$N), "tests"=sim_dataset$tests, "cases"=sim_dataset$cases)
ggplot(df_cts, aes(x=x, y=tests)) + geom_jitter(width=0.2, height=0, alpha=0.5)
ggplot(df_cts, aes(x=x, y=cases)) + geom_jitter(width=0.2, height=0, alpha=0.5)


## p4_model = cmdstan_model("binomial_4_preds.stan")
## p4_fit = p4_model$sample(data=sim_dataset, parallel_chains=4)
## p4_fit$diagnostic_summary()

## y_rep = p4_fit$summary("y_rep")$mean

## plot(sim_dataset$cases, y_rep, pch = 20)

