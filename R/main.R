source("R/grad_descent_logreg.R")
source("R/plot_federated_logreg.R")
source("R/data_generator.R")

## Federated Learning: Logistic Regression:
## =========================================================

library(tidyverse)
library(ggthemes)


# formula = as.formula("Species ~ Sepal.Length + Sepal.Width + Petal.Length")
# formula = as.formula("Species ~ Sepal.Length")
# X = model.matrix(formula, data = iris)
# y = ifelse(iris$Species == "setosa", 1, 0)

set.seed(3141)
data = simulateBinaryClassif(nrow = 4000, ncol = 2)

X = data$X
y = data$y

# IID Dataset
# ------------------------------------------

idx = sample(1:4, nrow(X), replace = TRUE)
index_list = lapply(1:4, function (x) { idx == x })

# Non-IID Dataset
# ------------------------------------------

# cat_var = cut(iris$Sepal.Width, breaks = c(-Inf, quantile(iris$Sepal.Width, c(0.33, 0.67)), Inf), labels = c(1,2,3))

# index_list = list(which(cat_var == 1), which(cat_var == 2), which(cat_var == 3))

# Fit models and create images:
# ------------------------------------------

vec_learning_rates = c(0.001, 0.01, 0.05, 0.1)

fed_logreg = getFittingTrace(X = X, y = y, index_list = index_list, max_iters = 2000L, 
  vec_iters_at_once = c(1, 10, 100, 1000, 2000), vec_learning_rates = vec_learning_rates, 
  true_beta = data$params[-1])

fed_logreg[[2]]