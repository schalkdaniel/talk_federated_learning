
sigmoid = function (X, beta) 
{
  return ((1 + exp(-X %*% beta))^(-1))
}
sigmoidGrad = function (X, beta) 
{
  sigmoid_temp = sigmoid(X, beta)
  return (sigmoid_temp * (1 - sigmoid_temp))
}

binomialLossFun = function (X, y, beta, activation = sigmoid)
{
  activate = activation(X, beta)
  return (- (y * log(activate) + (1 - y) * log(1 - activate)))
}
binomialLossGrad = function (X, y, beta, activation)
{
  return (X * (y - as.numeric(activation(X, beta))))
}

empiricalRisk = function (loss) 
{
  return (mean(loss))
}
empiricalRiskGrad = function (loss_grad)
{
  return (colMeans(loss_grad))
}

gradDescent = function (X, y, beta_start, learning_rate, epochs = 1000L, loss_grad_fun, activation_fun)
{
  for (iter in seq_len(epochs)) {
    loss_grad = loss_grad_fun(X = X, y = y, beta = beta_start, activation = activation_fun)
    beta_start = beta_start + learning_rate * empiricalRiskGrad(loss_grad)
  }
  return (beta_start)
}


## Gradient Descent Logistic Regression:
## =========================================================

# formula = as.formula("Species ~ Sepal.Length + Sepal.Width + Petal.Length")
formula = as.formula("Species ~ Sepal.Length")
X = model.matrix(formula, data = iris)
y = ifelse(iris$Species == "setosa", 1, 0)

beta = runif(ncol(X))
sigmoid(X, beta)
sigmoidGrad(X, beta)

binomialLossFun(X, y, beta)
loss_grad = binomialLossGrad(X, y, beta, activation = sigmoid)

empiricalRiskGrad(loss_grad)

mod = glm(y ~ 0 + X, family = binomial)
coef_glm = coef(mod)

coef_gd = rep(0, ncol(X))
for (i in seq_len(10)) {
  coef_gd = gradDescent(X = X, y = y, beta_start = coef_gd, learning_rate = 0.05, 
    epochs = 50000L, loss_grad = binomialLossGrad, activation = sigmoid)

  print(data.frame(glm = coef_glm, grad_desc = coef_gd))
  cat("\n\nRisk GLM:", empiricalRisk(binomialLossFun(X = X, y = y, beta = coef_glm, activation = sigmoid)))
  cat("\nRisk Grad Desc:", empiricalRisk(binomialLossFun(X = X, y = y, beta = coef_gd, activation = sigmoid)), "\n\n")
}

## Federated Learning: Logistic Regression:
## =========================================================

library(tidyverse)
library(ggthemes)

# IID Dataset
# ------------------------------------------

idx1 = sample(seq_len(nrow(X)), 50)
idx2 = sample(setdiff(seq_len(nrow(X)), idx1), 50)
idx3 = setdiff(seq_len(nrow(X)), c(idx1, idx2))

# Non-IID Dataset
# ------------------------------------------

cat_var = cut(iris$Sepal.Width, breaks = c(-Inf, quantile(iris$Sepal.Width, c(0.33, 0.67)), Inf), labels = c(1,2,3))

idx1 = which(cat_var == 1)
idx2 = which(cat_var == 2)
idx3 = which(cat_var == 3)

# Fit models and create images:
# ------------------------------------------

datasets = list(X1 = X[idx1, ], X2 = X[idx2, ], X3 = X[idx3, ])
responses = list(y1 = y[idx1], y2 = y[idx2], y3 = y[idx3])

max_iters = 200000L

vec_iters_at_once = c(1, 10, 100, 10000, 100000, 200000)
vec_learning_rates = c(0.001, 0.01, 0.05, 0.1)

printer_waits_iterations = 10000

data_plot_all = data.frame()

for (iters_at_once in vec_iters_at_once) {
  for (learning_rate in vec_learning_rates) {
    
    coef_gd = rep(0, ncol(X))
    global_model = list_coef
    list_coef = list(coef1 = coef_gd, coef2 = coef_gd, coef3 = coef_gd)

    iterations = seq_len(trunc(max_iters / iters_at_once) + 1)

    trace_coef = list(
      coef1 = matrix(NA, nrow = length(iterations) + 1, ncol = length(coef_gd)), 
      coef2 = matrix(NA, nrow = length(iterations) + 1, ncol = length(coef_gd)), 
      coef3 = matrix(NA, nrow = length(iterations) + 1, ncol = length(coef_gd)) 
    )
    
    trace_coef = lapply(trace_coef, function (x) { 
      x[1, ] = coef_gd 
      return (x)
    })

    for (i in iterations) {
      # Conduct gradient descent steps on each dataset:
      for(k in seq_along(list_coef)) {
        list_coef[[k]] = gradDescent(X = datasets[[k]], y = responses[[k]], beta_start = list_coef[[k]], learning_rate = learning_rate, 
          epochs = iters_at_once, loss_grad = binomialLossGrad, activation = sigmoid)
        trace_coef[[k]][i + 1, ] = list_coef[[k]]
      }
    
      # Apply Federated Averaging:
      global_model = unlist(apply(X = do.call(rbind, list_coef), MARGIN = 2, function (x) {
        weighted.mean(x = x, w = unlist(lapply(responses, length)))
      }))
    
      # Just the printer:
      if (i %% printer_waits_iterations == 0) {
        cat("Iteration:", i, ", Learning rate:", learning_rate, ", Iters at once:", iters_at_once, "\n\n")
        print(data.frame(glm = coef_glm, grad_desc = global_model, coef_x1 = list_coef[[1]], coef_x2 = list_coef[[2]], coef_x3 = list_coef[[3]]))
        cat("\n\nRisk GLM:", empiricalRisk(binomialLossFun(X = X, y = y, beta = coef_glm, activation = sigmoid)))
        cat("\nRisk Grad Desc:", empiricalRisk(binomialLossFun(X = X, y = y, beta = global_model, activation = sigmoid)), "\n\n")
      }
    }

  data_traces = lapply(trace_coef,  as.data.frame)
  for (i in seq_along(data_traces)) {
    data_traces[[i]]$set = paste0("Dataset", i)
  }
  data_traces = do.call(rbind, data_traces)
  data_traces$learning_rate = learning_rate
  data_traces$iters_at_once = iters_at_once
  names(data_traces) = c("Intercept", "Slope", "Dataset", "Learning Rate", "Iters at Once")
  
  # data_single_coef = as.data.frame(do.call(rbind, list_coef))
  # data_single_coef$Dataset = paste0("Dataset", 1:3)
  # names(data_single_coef) = c("Intercept", "Slope", "Dataset")

  data_plot_all = rbind(data_plot_all, data_traces)
  }
}

final_points = data_plot_all %>% group_by(Dataset, `Iters at Once`, `Learning Rate`) %>%
  filter(row_number() == n()) %>%
  group_by(`Iters at Once`, `Learning Rate`) %>%
  summarize(Intercept = mean(Intercept), Slope = mean(Slope))

data_all = data_plot_all %>% group_by(Dataset, `Iters at Once`, `Learning Rate`) %>%
  filter(row_number() %in% seq(1, n())) 

data_surface = expand.grid(
  Intercept = seq(min(data_all$Intercept), max(data_all$Intercept), length.out = 100), 
  Slope = seq(min(data_all$Slope), max(data_all$Slope), length.out = 100)
)
data_surface$risk = apply(data_surface, 1, function (params) { 
  empiricalRisk(
    binomialLossFun(X = X, y = y, beta = params, activation = sigmoid)
  ) 
})

glm_single = list()
for (i in seq_along(datasets)) {
  glm_single[[i]] = coef(glm(responses[[i]] ~ 0 + datasets[[i]], family = binomial))
  names(glm_single[[i]]) = c("feature1", "feature2") 
}
glm_single = as.data.frame(do.call(rbind, glm_single))
glm_single$dataset = paste0("Dataset", 1:3)

gg = ggplot() + 
  geom_raster(data = data_surface, aes(Intercept, Slope, fill = risk), show.legend = FALSE) +
  geom_contour(data = data_surface, aes(Intercept, Slope, z = risk), colour = "white", alpha = 0.5, size = 0.5) +
  geom_line(data = data_all, aes(x = Intercept, y = Slope, colour = Dataset), arrow = arrow(length=unit(0.20,"cm"), ends = "last", type = "closed")) + 
  geom_point(data = final_points, aes(x = Intercept, y = Slope), colour = rgb(238, 154, 0, 255, maxColorValue = 255), shape = 4, size = 2) +
  geom_text(data = final_points, aes(x = Intercept, y = Slope, label = "FedAvg"), colour = rgb(238, 154, 0, 255, maxColorValue = 255), vjust = -1) +
  annotate("point", x = coef_glm[1], y = coef_glm[2], colour = rgb(205, 150, 205, 255, maxColorValue = 255), shape = 4, size = 2) + 
  annotate("text", x = coef_glm[1], y = coef_glm[2], label = "GLM", vjust = -1, colour = rgb(205, 150, 205, 255, maxColorValue = 255)) + 
  geom_point(data = glm_single, aes(x = feature1, y = feature2, colour = dataset), show.legend = FALSE) +
  # geom_point(data = data_single_coef, aes(x = Intercept, y = Slope, colour = Dataset)) +
  theme_tufte() +
  scale_color_brewer(palette = "Spectral") +
  facet_grid(rows = vars(`Iters at Once`), cols = vars(`Learning Rate`)) +
    labs(title = "Centralized vs. Decentralized Learning", 
       subtitle = "Example using Logistic Regression and 200,000 iterations") +
  xlab("Feature 1") +
  ylab("Feature 2")

gg
