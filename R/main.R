source("R/grad_descent_logreg.R")
source("R/plot_federated_logreg.R")
source("R/data_generator.R")

## Federated Learning: Logistic Regression:
## =========================================================

library(tidyverse)
library(ggthemes)

# Setting 1: IID
# ------------------------------------------

set.seed(3141592)
data = simulateBinaryClassif(nrow = 4000, ncol = 2)

X = data$data
y = data$y

idx = sample(1:4, nrow(X), replace = TRUE)
index_list = lapply(1:4, function (x) { idx == x })

max_iters = 100000L
vec_learning_rates = c(0.001, 0.01, 0.05, 0.1)
vec_iters_at_once = c(1, 10, 100, 1000, 2000)
true_beta = data$params[-1]

fed_logreg_iid = getFittingTrace(X = X, y = y, index_list = index_list, max_iters = max_iters, 
  vec_iters_at_once = vec_iters_at_once, vec_learning_rates = vec_learning_rates, 
  true_beta = true_beta, eps_for_break = 1e-7)

plot(fed_logreg_iid, plot_title = "Federated Logistic Regression with Simulated IID Situation", show_averaging = TRUE)
# ggsave(filename = "images/fed_logreg_iid.pdf", width = 50, height = 30, units = "cm")



# Setting 2: Non-IID
# ------------------------------------------

# Situation: We now destroy the measurements of one feature of one dataset. This could correspond to 
#            a damaged sensor i.e. in a hospital. 

set.seed(3141592)
data = simulateBinaryClassif(nrow = 4000, ncol = 2)

X = data$data
y = data$y

idx = sample(1:4, nrow(X), replace = TRUE)
index_list = lapply(1:4, function (x) { idx == x })

# Destroy feature 2 of dataset 3:
idx_set = 3L
X[index_list[[idx_set]], 2] = X[index_list[[idx_set]], 2] + 4

max_iters = 100000L
vec_learning_rates = c(0.001, 0.01, 0.05, 0.1)
vec_iters_at_once = c(1, 10, 100, 1000, 2000)
true_beta = data$params[-1]

fed_logreg_set1 = getFittingTrace(X = X, y = y, index_list = index_list, max_iters = max_iters, 
  vec_iters_at_once = vec_iters_at_once, vec_learning_rates = vec_learning_rates, 
  true_beta = true_beta, eps_for_break = 1e-7, extract_global_model = extractGlobalModel)

plot(fed_logreg_set1, plot_title = "Federated Logistic Regression with Destroyed Feature of Dataset 3", show_averaging = TRUE)
# ggsave("images/fed_logreg_set1.pdf", width = 50, height = 30, units = "cm")



# Setting 3: IID Unbalanced
# ------------------------------------------

set.seed(3141592)
data = simulateBinaryClassif(nrow = 4000, ncol = 2)

X = data$data
y = data$y

idx = sample(1:4, nrow(X), replace = TRUE, prob = c(0.2, 0.1, 0.6, 0.1))
index_list = lapply(1:4, function (x) { idx == x })

max_iters = 100000L
vec_learning_rates = c(0.001, 0.01, 0.05, 0.1)
vec_iters_at_once = c(1, 10, 100, 1000, 2000)
true_beta = data$params[-1]

fed_logreg_iid_ub = getFittingTrace(X = X, y = y, index_list = index_list, max_iters = max_iters, 
  vec_iters_at_once = vec_iters_at_once, vec_learning_rates = vec_learning_rates, 
  true_beta = true_beta, eps_for_break = 1e-7)

plot(fed_logreg_iid_ub, plot_title = "Federated Logistic Regression with IID Setup", show_averaging = TRUE)
# ggsave(filename = "images/fed_logreg_iid_ub.pdf", width = 50, height = 30, units = "cm")


# Setting 4: Completely non-IID
# ------------------------------------------

set.seed(3141592)
data = simulateBinaryClassif(nrow = 4000, ncol = 2)

X = data$data
y = data$y

idx = sample(1:4, nrow(X), replace = TRUE, prob = c(0.2, 0.1, 0.6, 0.1))
index_list = lapply(1:4, function (x) { idx == x })

max_iters = 100000L
vec_learning_rates = c(0.001, 0.01, 0.05, 0.1)
vec_iters_at_once = c(1, 10, 100, 1000, 2000)
true_beta = data$params[-1]


biases_x1 = rnorm(length(index_list), mean = 0, sd = 1)
biases_x2 = rnorm(length(index_list), mean = 0, sd = 2)
for (idx_set in seq_along(index_list)) {
  X[index_list[[idx_set]], 1] = X[index_list[[idx_set]], 1] + biases_x1[idx_set]
  X[index_list[[idx_set]], 2] = X[index_list[[idx_set]], 2] + biases_x2[idx_set]
}

fed_logreg_ub_non_iid = getFittingTrace(X = X, y = y, index_list = index_list, max_iters = max_iters, 
  vec_iters_at_once = vec_iters_at_once, vec_learning_rates = vec_learning_rates, 
  true_beta = true_beta, eps_for_break = 1e-7)

plot(fed_logreg_ub_non_iid, plot_title = "Federated Logistic Regression with Unbalanced and Non-IID Setup", show_averaging = TRUE)
# ggsave(filename = "images/fed_logreg_ub_non_iid.pdf", width = 50, height = 30, units = "cm")
