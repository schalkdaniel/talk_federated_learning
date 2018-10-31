# ================================================================================= #
#                                                                                   #
#                                  Data Generator                                   #
#                                                                                   #
# ================================================================================= #

simulateBinaryClassif = function (nrow, ncol = 2, intercept = 0, betas) 
{
  if (missing(betas)) {
    #create beta distributed correlations
    corrs = rbeta(n = (ncol * (ncol - 1)) / 2, shape1 = 1, shape2 = 2)
    corrs = sample(c(-1, 1), size = length(corrs), replace = TRUE) * corrs

    sigma = matrix(1, nrow = ncol, ncol = ncol)
    sigma[upper.tri(sigma)] = corrs
    sigma[lower.tri(sigma)] = t(sigma)[lower.tri(sigma)]

    betas = c(intercept, runif(ncol, min = -2, max = 2))
  }
  data = as.data.frame(mvtnorm::rmvnorm(n = nrow, sigma = sigma, method = "svd"))

  y = rnorm(n = nrow, mean = as.matrix(cbind(1, data[,1:ncol])) %*% betas)
  y_binary = rbinom(n = nrow, 1, (1 + exp(-y))^(-1))

  if (intercept != 0) {
    data = cbind(Intercept = 1, data)
  }

  #return (list(data = data, betas = betas))
  return (list(data = data, y = y_binary, params = betas))
}

simulateBinaryClassifWithCategories = function (nrows, ncol = 2, categorical_bias = NULL)
{
  categorical_bias = unlist(categorical_bias)

  if (length(nrows) == 1) {
    nrows = rep(nrows, times = length(categorical_bias))
  }

  if (is.null(categorical_bias[1])) {
    stop("Cannot do anything.")
  }
  data_list = list(X = data.frame(), y = numeric(0), params = )

  for (i in seq_along(categorical_bias)) {
    data_list[[i]] = simulateBinaryClassif(nrows[i], ncol, categorical_bias[i])
    data_list[[i]]$data$category = paste0("set", i)
  }

  data_out = list()
  for (i in seq_along(categorical_bias)) {
    data_out[["X"]] = rbind(data_out[["X"]], data_list[[i]]$data)
    data_out[["y"]] = c(data_out[["y"]], data_list[[i]]$y)
    data_out[["params"]] = rbind(data_out[["params"]], data_list[[i]]$params)
  }
  return (do.call(rbind, data_list))
}

