# ================================================================================= #
#                                                                                   #
#                        Gradient Descent in Logistic Regression                    #
#                                                                                   #
# ================================================================================= #

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
